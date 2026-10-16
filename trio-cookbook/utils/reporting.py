"""Process-isolated SwanLab reporting shared by trainers.

SwanLab 0.10.1 permits one active run per process.  This utility provides a
small pipe-backed reporter for a secondary run without moving PyTRIO training
clients or model execution out of the trainer process.  It is deliberately
algorithm-agnostic: callers supply the run name and metadata.
"""

import json
import select
import subprocess
import sys
from dataclasses import asdict

from .metrics import MetricsAPI


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
            "    from trio_cookbook.utils.reporting import _worker\n"
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
            raise TimeoutError("Secondary SwanLab reporter did not respond.")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Secondary SwanLab reporter exited unexpectedly.")
        status, message = json.loads(line)
        if status != expected:
            raise RuntimeError(f"Secondary SwanLab reporter failed: {message}")

    def _send(self, action, payload):
        if self.closed or self.process.poll() is not None:
            raise RuntimeError("Secondary SwanLab reporter is not running.")
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


def isolated_metrics(config, *, name, metadata):
    """Create a metrics client backed by a separate SwanLab process.

    Reporting-disabled configurations return the normal no-op capable
    :class:`MetricsAPI`; enabled configurations use a pipe-backed secondary
    run.  The function does not know whether the run belongs to a critic or
    another trainer component.
    """

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


# Historical PPO callers used this name before reporting moved to utils.
critic_metrics = isolated_metrics
