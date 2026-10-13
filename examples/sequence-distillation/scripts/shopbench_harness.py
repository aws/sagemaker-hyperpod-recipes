"""Drive the ShoppingBench harness from a notebook: search server, agent rollouts, scoring, health checks.

ShoppingBench (https://github.com/yjwjy/ShoppingBench) pieces used here, all unmodified:
  src/search_engine/server.py   product search over the local Lucene index, on 127.0.0.1:5631 (port fixed in the tools)
  src/agent/run_rollout.py      the agent loop: prompt -> model reply -> parse <tool_call> -> run tool -> repeat
  src/agent/run_evaluate.py     the scoring functions (applied by score_rollouts.py in this folder)
Everything runs in the harness venv (sbenv) created by setup_shopbench.sh.
"""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import requests

SCRIPTS = Path(__file__).resolve().parent
SEARCH_URL = "http://127.0.0.1:5631"


def has_reasoning(text):
    """True if a reply carries actual reasoning. With thinking on, Qwen3.5's template opens <think> in the prompt, so
    the reply is "reasoning ... </think> answer". A fine-tuned model may also emit an empty "<think></think>" with
    thinking off, which does not count."""
    return "</think>" in text and bool(text.split("</think>")[0].replace("<think>", "").strip())


def nlines(p):
    return sum(1 for _ in open(p)) if Path(p).exists() else 0


def problem_queries(problems):
    return {json.loads(l)["query"] for l in open(problems) if l.strip()}


def trajectories(rollout, queries=None):
    """Trajectories in a rollout file, optionally only those for the given queries (a file can hold more, e.g. from
    an earlier run on another problem set)."""
    if not Path(rollout).exists():
        return []
    out = [json.loads(l) for l in open(rollout) if l.strip()]
    return out if queries is None else [t for t in out if t and t[0]["extra_info"]["query"] in queries]


class Harness:
    def __init__(self, work):
        """work: the folder setup_shopbench.sh populated (ShoppingBench/, sbenv/, jdk/)."""
        self.work = Path(work).expanduser().resolve()
        self.sb, self.py = self.work / "ShoppingBench", self.work / "sbenv" / "bin" / "python"
        self.logs = self.work / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        jdk = self.work / "jdk"
        java_home = jdk / "Contents" / "Home" if (jdk / "Contents" / "Home").exists() else jdk  # macOS vs Linux
        self.env = {**os.environ, "JAVA_HOME": str(java_home), "PATH": f"{java_home}/bin:" + os.environ["PATH"]}
        # pyserini 1.0.0 builds an OpenAI client at import time, and current openai releases raise without a key.
        # The harness passes its own key on every model call, so a placeholder is harmless.
        self.env.setdefault("OPENAI_API_KEY", "unused")
        self.procs = {}
        # Subprocesses are started detached (start_new_session=True) so a brief cell INTERRUPT doesn't kill
        # long rollouts. The downside: they also survive KERNEL SHUTDOWN/RESTART, orphaning the ~GB search
        # server on the port. Guard against that: on interpreter exit (kernel shutdown/restart) tear everything
        # down, and on SIGTERM (how Jupyter kills the kernel) do the same before the default handler runs.
        # NOTE: a plain cell interrupt (SIGINT->KeyboardInterrupt) does NOT exit the interpreter, so atexit does
        # not fire then — run_rollouts' own finally: stops rollouts in that case; the search server is kept alive
        # on purpose for reuse across rollout re-runs, and torn down here only when the kernel actually exits.
        self._cleanup_registered = False
        self._register_cleanup()

    def _register_cleanup(self):
        if self._cleanup_registered:
            return
        import atexit

        atexit.register(self.stop_all)
        try:
            prev = signal.getsignal(signal.SIGTERM)

            def _on_sigterm(signum, frame):
                self.stop_all()
                if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
                    prev(signum, frame)  # chain to any previously-installed handler
                else:
                    signal.signal(signal.SIGTERM, signal.SIG_DFL)
                    os.kill(os.getpid(), signal.SIGTERM)  # re-raise default behavior (terminate)

            signal.signal(signal.SIGTERM, _on_sigterm)
        except (ValueError, OSError):
            pass  # signal only settable in the main thread; atexit still covers clean shutdown
        self._cleanup_registered = True

    # ---------- background processes ----------
    def start(self, name, args, env=None):
        self.stop(name)
        log = open(self.logs / f"{name}.log", "w")
        self.procs[name] = subprocess.Popen(
            args,
            cwd=self.sb,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**self.env, **(env or {})},
            start_new_session=True,
        )
        return self.procs[name]

    def running(self, name):
        p = self.procs.get(name)
        return p is not None and p.poll() is None

    def stop(self, name):
        p = self.procs.pop(name, None)
        if p and p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)

    def stop_all(self):
        for name in list(self.procs):
            self.stop(name)

    # ---------- search server ----------
    @staticmethod
    def search_ok():
        try:
            return "product_id" in requests.get(f"{SEARCH_URL}/find_product?q=shoes&page=1", timeout=60).text
        except Exception:
            return False

    @staticmethod
    def _port_bound(port=5631):
        """True if something is bound to the local port (a server is holding it)."""
        import socket

        s = socket.socket()
        s.settimeout(2)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except Exception:
            return False
        finally:
            s.close()

    def _reclaim_search_port(self):
        """Kill any search-server process still holding port 5631 — e.g. one left running by a PREVIOUS kernel or
        session (self.stop only knows this harness's own children, so an orphan from another run isn't in
        self.procs). Without this, start_search would collide with 'Address already in use'. Dependency-free:
        finds the process by its command signature with pgrep (lsof/ss/psutil aren't available in sbenv)."""
        try:
            out = subprocess.run(["pgrep", "-f", "src/search_engine/server.py"], capture_output=True, text=True).stdout
        except Exception:
            return
        pids = [int(x) for x in out.split() if x.strip().isdigit() and int(x) != os.getpid()]
        for pid in pids:
            print(f"start_search: reclaiming port from stale search server (pid {pid})")
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    break
                for _ in range(10):  # up to ~5s per signal for it to exit
                    try:
                        os.kill(pid, 0)
                        time.sleep(0.5)
                    except ProcessLookupError:
                        break
                else:
                    continue  # still alive -> escalate to SIGKILL
                break

    def _search_argv(self, workers):
        """How to launch the search server. ShoppingBench's src/search_engine/server.py binds the Lucene index at
        import (module-level `searcher = LuceneSearcher("indexes")`) and exposes a module-level Flask `app`, and
        its tools hardcode http://127.0.0.1:5631 — so the index path is relative to cwd (=self.sb, where
        ShoppingBench/indexes lives) and the port must be 5631.

        PARALLELISM WITHOUT TOUCHING SHOPPINGBENCH: the stock server runs waitress with GIL-bound threads, so
        concurrent find_product calls (each a CPU-heavy k=100000 scan + per-doc JSON parse) serialize on one core
        — which is why queries time out under load even with CPU/RAM to spare. We instead host the SAME `app`
        under gunicorn with `workers` separate PROCESSES (own interpreter + own GIL each) for true parallelism.
        This imports server:app WITHOUT running its __main__ serve() block, so no ShoppingBench code is modified
        and search results are byte-identical — only the serving concurrency changes (benchmark fidelity intact).

        Each worker loads its own ~11 GB index copy (measured; mostly private anon heap, not shared on fork), so
        `workers` is capped by RAM, not CPU. --timeout exceeds the tools' 60s client timeout so a slow-but-valid
        query isn't worker-killed mid-request. Falls back to the stock single-process server if gunicorn is absent.
        """
        import shutil

        gunicorn = self.work / "sbenv" / "bin" / "gunicorn"
        gunicorn = str(gunicorn) if gunicorn.exists() else shutil.which("gunicorn")
        if gunicorn:
            # cwd stays self.sb (set by start()), so LuceneSearcher("indexes") resolves; PYTHONPATH lets
            # "server:app" import from src/search_engine without changing cwd.
            return [
                gunicorn,
                "--workers",
                str(workers),
                "--bind",
                "127.0.0.1:5631",
                "--timeout",
                "300",
                "--graceful-timeout",
                "30",
                "server:app",
            ], {"PYTHONPATH": str(self.sb / "src" / "search_engine")}
        print(
            "start_search: gunicorn not found in sbenv; falling back to single-process server "
            "(install with: sbenv/bin/pip install gunicorn — then re-run for parallel workers)."
        )
        return [str(self.py), "src/search_engine/server.py"], {"HOST": "127.0.0.1", "PORT": "5631"}

    def start_search(self, workers=12):
        """Start the search server unless a HEALTHY one already answers. Loading the index takes a few minutes
        (and multiplies with `workers`, since each worker loads its own index copy, ~4.2 GB resident).

        Sizing: find_product is single-threaded CPU per query (k=100000 scan + per-doc JSON parse, ~16s each,
        GIL-bound so one query = one core). Throughput scales with worker PROCESSES up to the CORE count (16
        here), NOT with RAM. Default 12 uses ~12 cores for search (~50 GB RAM) and leaves headroom for rollout
        procs; measured ~4.2 GB/worker so 12 fits in ~62 GB with buffer. If timeouts/queue-depth reappear, lower
        rollout threads (demand) before touching workers (supply).

        `workers`: number of gunicorn worker PROCESSES hosting ShoppingBench's Flask app (see _search_argv for
        why processes, not threads). Each loads its own index copy (~4.5 GB resident, measured). Default 8 fits in
        ~36 GB; with 16 cores you can push toward 10-12 if RAM allows. ShoppingBench code is never modified —
        results are unchanged.

        Self-healing: if the health check fails but the port is still held by a stale server (e.g. left running by
        a previous kernel/session that never ran cleanup), that orphan is killed first so the fresh start doesn't
        collide with 'Address already in use'. If rollouts log many 'Read timed out' errors the server is
        overloaded or stuck — re-run this cell (optionally with more workers) to restart it."""
        if self.search_ok():
            return  # a healthy server is already up; reuse it
        if self._port_bound():
            self._reclaim_search_port()  # stale/wedged server holding the port -> reclaim it
        argv, env = self._search_argv(workers)
        self.start("search", argv, env=env)
        for _ in range(360):
            if self.search_ok():
                return
            if not self.running("search"):
                raise RuntimeError(f"search server exited; see {self.logs}/search.log")
            time.sleep(10)
        raise TimeoutError("search server did not come up")

    # ---------- rollouts ----------
    def write_rollout_cfg(
        self, out_dir, tag, task, problems, base_url, enable_thinking, threads=4, temperature=0, max_tokens=4096
    ):
        """run_rollout.py's config. Notes:
        - model "" : the endpoint is chosen by URL; vLLM accepts an empty model name.
        - thinking goes in chat_template_kwargs; a top-level {"enable_thinking": ...} (as in the upstream example
          configs) is silently ignored by vLLM.
        - max_tokens is per step (one reply), not per trajectory. Kept at 4096 (not 8192): the model's context
          is 40960, and vLLM validates prompt_tokens + max_tokens <= 40960, rejecting the request with a 4XX
          (424 VLLMValidationError) otherwise. On long trajectories the accumulated prompt grows large, so an
          8192 output budget overflowed (prompt 32769 + 8192 = 40961 > 40960) and failed those steps. 4096
          leaves headroom for prompts up to 36864 tokens; a tool-call reply (thinking off) is far shorter."""
        out_dir = Path(out_dir)
        rollout = out_dir.resolve() / f"rollout_{tag}_{task}.jsonl"
        cfg = {
            "task": task,
            "system_prompt_file": "src/agent/prompt/rollout.md",
            "synthesize_file": str(Path(problems).resolve()),
            "rollout_file": str(rollout),
            "threads": threads,
            "base_url": base_url,
            "api_key": "unused",
            "model_config": {
                "model": "",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            },
        }
        path = out_dir / f"cfg_{tag}_{task}.json"
        path.write_text(json.dumps(cfg, indent=2))
        return path, rollout

    def run_rollouts(self, jobs, poll_s=120):
        """jobs: {name: (cfg_path, rollout_path, n_problems)}. All jobs run concurrently; blocks until done.
        Resume-safe: run_rollout.py skips problems already in the rollout file, so an interrupted run continues
        where it stopped (interrupting this cell also stops the trajectories). Progress counts only trajectories for
        the job's own problems.

        Reporting: each poll refreshes an aligned per-job table IN PLACE (no scrolling history) — tag, a text
        progress bar, done/total, percent, the number completed since the last poll (throughput), and run/done
        status — so 9 concurrent jobs are readable at a glance. Falls back to plain prints outside IPython.
        """
        queries = {name: problem_queries(json.load(open(cfg))["synthesize_file"]) for name, (cfg, _, _) in jobs.items()}

        try:
            from IPython.display import clear_output
        except Exception:
            clear_output = None

        def done_count(name):
            _, r, _ = jobs[name]
            return len({t[0]["extra_info"]["query"] for t in trajectories(r, queries[name])})

        for name, (cfg, rollout, n) in jobs.items():
            if len(trajectories(rollout, queries[name])) < n:
                self.start(f"rollout_{name}", [str(self.py), "src/agent/run_rollout.py", str(cfg)])

        def report(prev):
            width = max((len(name) for name in jobs), default=4)
            cur = {}
            total_done = total_n = 0
            lines = []
            for name, (_, _, n) in jobs.items():
                d = done_count(name)
                cur[name] = d
                total_done += d
                total_n += n
                frac = d / n if n else 1.0
                filled = int(round(frac * 20))
                bar = "█" * filled + "·" * (20 - filled)
                delta = d - prev.get(name, d)
                alive = self.running(f"rollout_{name}")
                status = "run " if alive else ("done" if d >= n else "stop")
                rate = f"+{delta}" if delta else "  ·"
                lines.append(f"  {name:<{width}}  [{bar}] {d:>4}/{n:<4} {frac*100:5.1f}%  {rate:>4}/poll  {status}")
            agg = total_done / total_n if total_n else 1.0
            header = (
                f"[{time.strftime('%H:%M:%S')}] rollouts {total_done}/{total_n} "
                f"({agg*100:.1f}%) — {sum(self.running(f'rollout_{nm}') for nm in jobs)}/{len(jobs)} jobs running"
            )
            if clear_output:
                clear_output(wait=True)  # overwrite the previous table (no history)
            print(header + "\n" + "\n".join(lines), flush=True)
            return cur

        prev = {name: 0 for name in jobs}
        try:
            while any(self.running(f"rollout_{name}") for name in jobs):
                prev = report(prev)
                time.sleep(poll_s)
        finally:
            for name in jobs:
                self.stop(f"rollout_{name}")
        report(prev)  # final table once all jobs have exited

    # ---------- scoring ----------
    def score(self, task, rollout, problems):
        """ShoppingBench's own rules via score_rollouts.py -> {"n", "asr", "exact_id", "recommend"}."""
        r = subprocess.run(
            [
                str(self.py),
                str(SCRIPTS / "score_rollouts.py"),
                "--task",
                task,
                "--rollouts",
                str(Path(rollout).resolve()),
                "--problems",
                str(Path(problems).resolve()),
            ],
            cwd=self.sb,
            env=self.env,
            capture_output=True,
            text=True,
        )
        if r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    @staticmethod
    def health(rollout, problems=None):
        """Run-quality checks (the same ones ShoppingBench's run.sh prints), as shares of trajectories:
        empty_content       model call returned nothing (endpoint errors, context overflow)
        null_tool_results   a tool call failed (usually the search server timing out)
        no_terminate        trajectory ended without the terminate tool: mostly a reply the harness could not parse
        steps_with_reasoning  share of steps with a non-empty <think> block (~1 with thinking on, ~0 off)
        problems: count only trajectories for these problems (default: all in the file)."""
        trajs = trajectories(rollout, problem_queries(problems) if problems else None)
        lines = [json.dumps(t, ensure_ascii=False, separators=(",", ":")) for t in trajs]
        n = max(len(lines), 1)
        steps = [s for t in trajs for s in t]
        return {
            "empty_content": round(sum('"reasoning_content":"","content":""' in l for l in lines) / n, 3),
            "null_tool_results": round(sum('"results":null' in l for l in lines) / n, 3),
            "no_terminate": round(sum('"name":"terminate"' not in l for l in lines) / n, 3),
            "steps_with_reasoning": round(
                sum(has_reasoning(s["completion"].get("content") or "") for s in steps) / max(len(steps), 1), 3
            ),
        }
