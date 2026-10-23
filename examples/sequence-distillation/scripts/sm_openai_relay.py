"""Local OpenAI-compatible relay in front of SageMaker AI endpoints (and Bedrock models).

ShoppingBench and lm-eval create an OpenAI client from a fixed base_url / api_key. This relay is that local
OpenAI endpoint: each one listens on 127.0.0.1:<port>, takes a Chat Completions request, and forwards it to a
SageMaker endpoint via sagemaker-runtime InvokeEndpoint (SigV4 per call, so multi-hour runs never fail on token
expiry). InvokeEndpoint is used — rather than the endpoint's /openai/v1 passthrough URL — because routing to a
specific *inference component* (base vs a LoRA adapter) is only honored via InvokeEndpoint's InferenceComponentName
(the OpenAI route rejects IC endpoints). The reply is streamed (InvokeEndpointWithResponseStream: plain InvokeEndpoint
times out after 60 s, which long thinking steps exceed) and reassembled into one Chat Completions body. With thinking
on, the opening <think> the chat template put in the prompt is restored in the reply, so ShoppingBench keeps the
reasoning in the next step's history (as in the training data). A relay can also
  - force fields into every request (force={"chat_template_kwargs": {"enable_thinking": False}} for IFEval), and
  - send the request to a Bedrock model through the Converse API instead (transport="bedrock"), so a frontier model
    without an OpenAI-compatible endpoint runs through the same harness.

    url = start_relay("base", "my-endpoint", 8101, region="us-west-2", inference_component="base-ic")
"""
import http.server
import json
import threading
import time

import boto3
from botocore.config import Config

RELAYS = {}


def _call_sagemaker_openai(region, endpoint, body):
    """Single-model endpoints: the endpoint's OpenAI-compatible route with a short-lived bearer token (as in the
    sagemaker-genai-pr relay). DJL/LMI's /invocations chat parser is broken in lmi25 ("No module named
    vllm.entrypoints.openai.protocol"), so InvokeEndpoint is only used for inference components."""
    import requests
    from sagemaker.core.token_generator import generate_token

    url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint}/openai/v1/chat/completions"
    body["model"] = ""
    for attempt in range(6):
        try:
            r = requests.post(
                url, json=body, timeout=900, headers={"Authorization": f"Bearer {generate_token(region=region)}"}
            )
        except Exception as e:
            status, content = 502, json.dumps({"error": {"message": str(e)[:500]}}).encode()
            time.sleep(2**attempt)
            continue
        status, content = r.status_code, r.content
        if status not in (429, 500, 502, 503, 504):
            break
        time.sleep(2**attempt)
    return status, content


def _call_sagemaker(region, endpoint, body, inference_component=None):
    """Forward a Chat Completions request to an inference component via InvokeEndpoint(WithResponseStream).
    Returns (status_code, response_bytes) with an OpenAI-shaped chat.completion body."""
    if not inference_component:
        return _call_sagemaker_openai(region, endpoint, body)
    rt = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=Config(read_timeout=900, retries={"max_attempts": 6, "mode": "adaptive"}),
    )
    # Stream the reply: plain InvokeEndpoint times out after 60 s, which long thinking steps over a long history
    # exceed (~63 s observed) -- and greedy decoding makes every retry time out the same way. The streaming call
    # has no such limit; the chunks are reassembled into the same chat.completion body the caller expects.
    sbody = dict(body, stream=True, stream_options={"include_usage": True})
    kwargs = dict(EndpointName=endpoint, ContentType="application/json", Body=json.dumps(sbody))
    if inference_component:
        kwargs["InferenceComponentName"] = inference_component  # boto3 -> X-Amzn-SageMaker-Inference-Component
    if inference_component not in _NO_STREAM:
        try:
            srt = boto3.client(
                "sagemaker-runtime",
                region_name=region,
                config=Config(read_timeout=120, retries={"max_attempts": 2, "mode": "adaptive"}),
            )
            r = srt.invoke_endpoint_with_response_stream(**kwargs)
            buf = b""
            for ev in r["Body"]:
                if "PayloadPart" in ev:
                    buf += ev["PayloadPart"]["Bytes"]
                elif ev:  # ModelStreamError / InternalStreamFailure
                    raise RuntimeError(json.dumps(ev, default=str)[:2000])
            return 200, json.dumps(_assemble_stream(buf.decode())).encode()
        except Exception as e:
            if "Read timeout" not in str(e):
                return 500, json.dumps({"error": {"message": str(e)[:2000], "type": "relay_invoke_error"}}).encode()
            # Streaming to some adapter ICs hangs (no bytes ever arrive); use plain InvokeEndpoint for this IC from now on
            _NO_STREAM.add(inference_component)
    kwargs["Body"] = json.dumps(body)
    try:
        out = json.loads(rt.invoke_endpoint(**kwargs)["Body"].read())
        for ch in out.get("choices") or []:
            msg = ch.get("message") or {}
            msg["content"] = _restore_think(msg.get("content") or "")
        return 200, json.dumps(out).encode()
    except Exception as e:
        return 500, json.dumps({"error": {"message": str(e)[:2000], "type": "relay_invoke_error"}}).encode()


_NO_STREAM = set()


def _restore_think(text):
    """Thinking on: the chat template already opened <think> in the prompt, so the reply carries only the closing
    tag. Restore the opening tag so ShoppingBench parses the reasoning and keeps it in the next step's history
    (Message.from_string needs <think>...</think> or reasoning_content), as in training and the teacher runs."""
    if "</think>" in text and "<think>" not in text.split("</think>", 1)[0]:
        return "<think>\n" + text.lstrip("\n")
    return text


def _assemble_stream(text):
    """Newline-delimited chat.completion.chunk JSON (optionally SSE 'data: ' lines) -> one chat.completion."""
    content, reasoning, finish, usage, head = [], [], None, None, {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        chunk = json.loads(line)
        if "error" in chunk:
            raise RuntimeError(json.dumps(chunk["error"])[:2000])
        head = head or {k: chunk.get(k) for k in ("id", "created", "model")}
        usage = chunk.get("usage") or usage
        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            finish = ch.get("finish_reason") or finish
    message = {"role": "assistant", "content": _restore_think("".join(content))}
    if any(reasoning):
        message["reasoning_content"] = "".join(reasoning)
    return {
        **head,
        "object": "chat.completion",
        "usage": usage,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }


def _strip_think(text):
    """Remove every <think>...</think> block (and the whitespace after it). A linear scan rather than a regex:
    the content comes from the request body, and a lazy-dot pattern is quadratic on many unclosed <think> tags."""
    out, i = [], 0
    while True:
        start = text.find("<think>", i)
        end = text.find("</think>", start + 7) if start >= 0 else -1
        if end < 0:
            out.append(text[i:])
            return "".join(out)
        out.append(text[i:start])
        i = end + len("</think>")
        while i < len(text) and text[i].isspace():
            i += 1


def _call_bedrock(region, model_id, body, extra_fields=None, keep_reasoning=True):
    """Chat Completions -> Bedrock Converse -> Chat Completions. A fresh client per call picks up refreshed
    credentials during long runs. The model's reasoning (Converse reasoningContent blocks) is returned as a
    leading <think>...</think> in the content, the same shape a Qwen thinking model produces; earlier turns'
    reasoning is stripped before the history is sent back. extra_fields goes to additionalModelRequestFields
    (e.g. {"reasoning_effort": "high"} for gpt-oss)."""
    # Fail fast on a stuck call: 300 s covers the longest legitimate replies (8192 tokens, ~2-4 min), and 3 attempts
    # keep a hung connection from blocking a rollout worker for hours. A failed step ends that trajectory unsolved,
    # so rejection sampling simply drops it.
    br = boto3.Session().client(
        "bedrock-runtime",
        region_name=region,
        config=Config(read_timeout=300, retries={"max_attempts": 3, "mode": "adaptive"}),
    )
    msgs = body["messages"]
    system = [{"text": m["content"]} for m in msgs if m["role"] == "system"]
    conv = [
        {"role": m["role"], "content": [{"text": _strip_think(m["content"]) or " "}]}
        for m in msgs
        if m["role"] in ("user", "assistant")
    ]

    def _converse(include_temperature):
        cfg = {"maxTokens": int(body.get("max_tokens", 4096))}
        if include_temperature:
            cfg["temperature"] = float(body.get("temperature", 0))
            if body.get("top_p") is not None:
                cfg["topP"] = float(body["top_p"])
        kw = {"additionalModelRequestFields": extra_fields} if extra_fields else {}
        return br.converse(modelId=model_id, system=system, messages=conv, inferenceConfig=cfg, **kw)

    try:
        try:
            r = _converse(include_temperature=True)
        except br.exceptions.ValidationException as ve:
            # Some newer models deprecate/reject `temperature` with a ValidationException.
            # Retry without it so the relay stays model-agnostic. NOTE: such models run at their own default
            # temperature (we can't force 0), a minor determinism caveat for those references only.
            if "temperature" in str(ve).lower():
                r = _converse(include_temperature=False)
            else:
                raise
        blocks = r["output"]["message"]["content"]
        text = "".join(b.get("text", "") for b in blocks)
        # Collect by block type: models differ in order (gpt-oss: reasoning first; GLM: text first).
        reasoning = "".join(
            b["reasoningContent"].get("reasoningText", {}).get("text", "") for b in blocks if "reasoningContent" in b
        )
        if reasoning and keep_reasoning:
            text = f"<think>\n{reasoning.strip()}\n</think>\n\n{text.lstrip()}"
        out = {
            "id": "bedrock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length" if r["stopReason"] == "max_tokens" else "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {
                "prompt_tokens": r["usage"]["inputTokens"],
                "completion_tokens": r["usage"]["outputTokens"],
                "total_tokens": r["usage"]["totalTokens"],
            },
        }
        return 200, json.dumps(out).encode()
    except Exception as e:
        return 500, json.dumps({"error": str(e)[:2000]}).encode()


def _call_http(base_url, body, model):
    """transport="http": forward to a plain OpenAI-compatible server (e.g. vLLM behind a port-forward)."""
    import requests

    body["model"] = model or body.get("model", "")
    for attempt in range(6):
        try:
            r = requests.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=1800)
            if r.status_code not in (429, 500, 502, 503, 504):
                return r.status_code, r.content
        except Exception as e:
            r = None
            err = str(e)[:500]
        time.sleep(2**attempt)
    return (r.status_code, r.content) if r is not None else (502, json.dumps({"error": {"message": err}}).encode())


class _Relay(http.server.BaseHTTPRequestHandler):
    region, target, force, transport, inference_component, extra_fields, keep_reasoning = (
        None,
        None,
        None,
        "sagemaker",
        None,
        None,
        True,
    )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        body["stream"] = False
        body.update(self.force or {})
        if self.transport == "bedrock":
            status, content = _call_bedrock(self.region, self.target, body, self.extra_fields, self.keep_reasoning)
        elif self.transport == "http":
            status, content = _call_http(
                self.target, body, self.inference_component
            )  # inference_component = model name
        else:
            # The LMI/vLLM container selects the model by endpoint + InferenceComponentName, not by the OpenAI
            # "model" field — and it 404s ("model `X` does not exist") on an unrecognized name. Clients vary:
            # the agent harness sends model="" (accepted), but lm-eval sends model="eval" (rejected -> 500 ->
            # the whole eval cascades to failure). Blank it so routing is purely by IC regardless of client.
            body["model"] = ""
            status, content = _call_sagemaker(self.region, self.target, body, self.inference_component)
        # The client (e.g. lm-eval under high num_concurrent) may disconnect before/while we respond — tearing
        # down its aiohttp session cancels in-flight requests. Guard the write so a dropped connection doesn't
        # raise BrokenPipeError/ConnectionResetError out of the handler thread (which spams tracebacks and, with
        # lm-eval, can cascade the whole eval to fail). A disconnected client simply gets no response.
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


def start_relay(
    name,
    target,
    port,
    region,
    force=None,
    transport="sagemaker",
    inference_component=None,
    extra_fields=None,
    keep_reasoning=True,
):
    """target: an endpoint name (transport="sagemaker") or a Bedrock model id (transport="bedrock").
    inference_component: for transport="sagemaker", the inference component to address on the endpoint
    (sent as the OpenAI "model" field); None routes to the single model behind the endpoint.
    extra_fields: for transport="bedrock", passed to Converse as additionalModelRequestFields.
    keep_reasoning: for transport="bedrock", prepend the model's reasoning as <think>...</think> (False for IFEval).
    Starting a relay under an existing name replaces it. Returns the base_url for an OpenAI client."""
    stop_relay(name)
    handler = type(
        f"Relay_{name}",
        (_Relay,),
        {
            "region": region,
            "target": target,
            "force": force,
            "transport": transport,
            "inference_component": inference_component,
            "extra_fields": extra_fields,
            "keep_reasoning": keep_reasoning,
        },
    )
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    RELAYS[name] = srv
    return f"http://127.0.0.1:{port}/v1"


def stop_relay(name):
    srv = RELAYS.pop(name, None)
    if srv:
        srv.shutdown()
        srv.server_close()


def stop_all():
    for name in list(RELAYS):
        stop_relay(name)
