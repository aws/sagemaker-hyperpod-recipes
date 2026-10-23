"""Small cross-cutting helpers shared by the distillation/eval notebooks (no SageMaker coupling)."""
import subprocess


def run(*args, **kw):
    """Run a command and stream its combined stdout/stderr line-by-line (so it shows live in a notebook).

    args : the command and its arguments; each is str()-ified, so Path objects are fine.
    kw   : forwarded to subprocess.Popen (e.g. env=, cwd=).
    Raises RuntimeError with the exit code and command if the process exits non-zero.
    """
    p = subprocess.Popen([str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    for line in p.stdout:
        print(line, end="")
    if p.wait():
        raise RuntimeError(f"exit code {p.returncode}: {args}")
