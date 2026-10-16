"""A second SwanLab run in its own process (0.10.1 allows one per process).

Only metric transport is isolated. Both PyTRIO training clients stay in the
trainer process. A module subprocess also works for notebooks and stdin scripts.
"""

import json
import select
import subprocess
import sys
from dataclasses import asdict

from ...utils.metrics import MetricsAPI


def _worker(source, sink):
    import swanlab

    run = None
    try:
        for line in source:
            action, payload = json.loads(line)
            if action == "init":
                run = swanlab.init(**payload)
                sink.write(json.dumps(["ready", None]) + "\n")
                sink.flush()
            elif action == "finish":
                run.finish(**payload)
                sink.write(json.dumps(["finished", None]) + "\n")
                sink.flush()
                return
            else:
                getattr(run, action)(**payload)
        if run is not None:
            run.finish(
                state="crashed", error="Parent reporter pipe closed before finish."
            )
    except BaseException as error:
        sink.write(json.dumps(["error", f"{type(error).__name__}: {error}"]) + "\n")
        sink.flush()
        if run is not None:
            run.finish(state="crashed", error=str(error))


class _RunProcess:
    def __init__(self, kwargs, *, timeout=90):
        self.timeout = timeout
        self.closed = False
        code = (
            "import sys\nfrom contextlib import redirect_stdout\n"
            "with redirect_stdout(sys.stderr):\n"
            "    from trio_cookbook.trainer.ppo.reporting import _worker\n"
            "    _worker(sys.__stdin__, sys.__stdout__)\n"
        )
        self.process = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            self._send("init", kwargs)
            self._wait("ready")
        except BaseException:
            self._cleanup()
            raise

    def _wait(self, expected):
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            raise TimeoutError("Critic SwanLab reporter did not respond.")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Critic SwanLab reporter exited unexpectedly.")
        status, message = json.loads(line)
        if status != expected:
            raise RuntimeError(f"Critic SwanLab reporter failed: {message}")

    def _send(self, action, payload):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("Critic SwanLab reporter is not running.")
        if select.select([self.process.stdout], [], [], 0)[0]:
            self._wait("running")  # Only unexpected failures arrive between commands.
        self.process.stdin.write(json.dumps([action, payload], allow_nan=False) + "\n")
        self.process.stdin.flush()

    def log(self, data, *, step):
        self._send("log", dict(data=data, step=step))

    def log_text(self, *, key, data, step):
        self._send("log_text", dict(key=key, data=data, step=step))

    def _cleanup(self):
        self.closed = True
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process.stdout.close()

    def finish(self, *, state="success", error=None):
        if self.closed:
            return
        try:
            self._send("finish", dict(state=state, error=error))
            self._wait("finished")
        finally:
            self._cleanup()


def critic_metrics(config, *, name, metadata):
    if config.report_to is None:
        return MetricsAPI()
    return MetricsAPI(
        _RunProcess(
            dict(
                project=config.project,
                workspace=config.workspace,
                tags=config.tags,
                name=name,
                log_dir=config.log_dir,
                config={**asdict(config), **metadata},
            )
        )
    )
