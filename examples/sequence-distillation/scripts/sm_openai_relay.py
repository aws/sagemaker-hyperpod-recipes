"""Local OpenAI-compatible relay in front of SageMaker AI endpoints (and Bedrock models).

ShoppingBench and lm-eval create an OpenAI client from a fixed base_url / api_key. This relay is that local
OpenAI endpoint: each one listens on 127.0.0.1:<port>, takes a Chat Completions request, and forwards it to a
SageMaker endpoint via sagemaker-runtime InvokeEndpoint (SigV4 per call, so multi-hour runs never fail on token
expiry). InvokeEndpoint is used — rather than the endpoint's /openai/v1 passthrough URL — because routing to a
specific *inference component* (base vs a LoRA adapter) is only honored via InvokeEndpoint's InferenceComponentName
(the OpenAI route rejects IC endpoints). The LMI/vLLM container returns a Chat-Completions-shaped body on
/invocations, so the relay forwards it to the OpenAI client unchanged. A relay can also
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


def _call_sagemaker(region, endpoint, body, inference_component=None):
    """Forward a Chat Completions request to a SageMaker endpoint via InvokeEndpoint, routing to a specific
    inference component when given. Returns (status_code, response_bytes). The LMI container answers on
    /invocations with an OpenAI-shaped chat.completion body, so we pass it straight back to the caller."""
    rt = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=Config(read_timeout=900, retries={"max_attempts": 6, "mode": "adaptive"}),
    )
    kwargs = dict(
        EndpointName=endpoint, ContentType="application/json", Accept="application/json", Body=json.dumps(body)
    )
    if inference_component:
        kwargs["InferenceComponentName"] = inference_component  # boto3 -> X-Amzn-SageMaker-Inference-Component
    try:
        r = rt.invoke_endpoint(**kwargs)
        return 200, r["Body"].read()
    except Exception as e:
        return 500, json.dumps({"error": {"message": str(e)[:2000], "type": "relay_invoke_error"}}).encode()


def _call_bedrock(region, model_id, body):
    """Chat Completions -> Bedrock Converse -> Chat Completions. A fresh client per call picks up refreshed
    credentials during long runs."""
    br = boto3.Session().client(
        "bedrock-runtime",
        region_name=region,
        config=Config(read_timeout=900, retries={"max_attempts": 8, "mode": "adaptive"}),
    )
    msgs = body["messages"]
    system = [{"text": m["content"]} for m in msgs if m["role"] == "system"]
    conv = [
        {"role": m["role"], "content": [{"text": m["content"]}]} for m in msgs if m["role"] in ("user", "assistant")
    ]

    def _converse(include_temperature):
        cfg = {"maxTokens": int(body.get("max_tokens", 4096))}
        if include_temperature:
            cfg["temperature"] = float(body.get("temperature", 0))
        return br.converse(modelId=model_id, system=system, messages=conv, inferenceConfig=cfg)

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
        text = "".join(b.get("text", "") for b in r["output"]["message"]["content"])
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
        }
        return 200, json.dumps(out).encode()
    except Exception as e:
        return 500, json.dumps({"error": str(e)[:2000]}).encode()


class _Relay(http.server.BaseHTTPRequestHandler):
    region, target, force, transport, inference_component = None, None, None, "sagemaker", None

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        body["stream"] = False
        body.update(self.force or {})
        if self.transport == "bedrock":
            status, content = _call_bedrock(self.region, self.target, body)
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


def start_relay(name, target, port, region, force=None, transport="sagemaker", inference_component=None):
    """target: an endpoint name (transport="sagemaker") or a Bedrock model id (transport="bedrock").
    inference_component: for transport="sagemaker", the inference component to address on the endpoint
    (sent as the OpenAI "model" field); None routes to the single model behind the endpoint.
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
