"""Cross-cutting checkpoint names and validated sync/async saves.

Save cadence and state publication belong to each algorithm. In particular,
SFT submits queued saves in its trainer to preserve the SDK's request ordering.
"""

import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..api.client_api import call, call_async
from .dataloader import dataset_identity, training_rows
from .scheduler import optimizer_horizons, scheduler_state


def checkpoint_name(algorithm: str, step: int) -> str:
    return f"{algorithm}-step-{step:08d}"


def checkpoint_path(saved: Any, *, method: str = "save_state") -> str:
    path = getattr(saved, "path", None)
    if not isinstance(path, str) or not path:
        raise TypeError(f"PyTRIO {method}() returned no checkpoint path.")
    return path


def save_checkpoint(client: Any, algorithm: str, step: int) -> str:
    saved = call(client, "save_state", name=checkpoint_name(algorithm, step))
    return checkpoint_path(saved)


async def save_checkpoint_async(
    client: Any,
    algorithm: str,
    step: int,
    *,
    error_method: str = "save_state_async",
) -> str:
    saved = await call_async(
        client, "save_state", name=checkpoint_name(algorithm, step)
    )
    return checkpoint_path(saved, method=error_method)


# ---------------------------------------------------------------------------
# Versioned local runtime manifests
# ---------------------------------------------------------------------------


FORMAT_VERSION = 2
_RUNTIME_FIELDS = (
    "_last_eval_step",
    "_train_tokens",
    "_total_tokens",
    "_total_pairs",
    "_sample_input_tokens",
    "_sample_output_tokens",
    "_eval_sample_input_tokens",
    "_eval_sample_output_tokens",
    "_next_group",
    "_sampling_seed_offset",
)


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _tuple_tree(value):
    return tuple(_tuple_tree(v) for v in value) if isinstance(value, list) else value


def capture_rng():
    """Serialize local RNGs as plain JSON, never executable pickle payloads.

    PyTRIO owns remote optimizer state and accepts explicit sampling seeds.
    Python/NumPy/Torch CPU state covers local reward/loss randomness; sampling
    seed offsets are retained separately in trainer/agent runtime state.
    """
    import numpy as np

    np_state = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
    }
    if "torch" in sys.modules:
        import torch

        result["torch_cpu"] = torch.get_rng_state().tolist()
        if torch.cuda.is_initialized():
            result["torch_cuda"] = [
                state.tolist() for state in torch.cuda.get_rng_state_all()
            ]
    return result


def restore_rng(value):
    """Restore only known RNG formats after clients and hooks are initialized."""
    import numpy as np

    random.setstate(_tuple_tree(value["python"]))
    algorithm, keys, position, has_gauss, cached = value["numpy"]
    np.random.set_state(
        (algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached)
    )
    if "torch_cpu" in value:
        import torch

        torch.set_rng_state(torch.tensor(value["torch_cpu"], dtype=torch.uint8))
        if "torch_cuda" in value:
            torch.cuda.set_rng_state_all(
                [torch.tensor(v, dtype=torch.uint8) for v in value["torch_cuda"]]
            )


def load_training_checkpoint(path):
    """Read the unified manifest before allocating any remote training client.

    Only the versioned JSON format is accepted. A bare PyTRIO URI has no cursor,
    optimizer identity or RNG progress and cannot provide complete continuation.
    Trainers validate its configuration/data identity before using its URIs.
    """
    if str(path).startswith("trio://"):
        raise ValueError(
            "Complete resume requires a local JSON manifest, not a bare weight URI."
        )
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        payload = json.load(stream)
    required = {
        "format_version",
        "algorithm",
        "identity",
        "identity_sha256",
        "components",
        "state",
        "runtime",
        "rng",
        "loop",
        "reporting",
    }
    if isinstance(payload, dict) and payload.get("format_version") == FORMAT_VERSION:
        required.add("schedulers")
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or type(payload["format_version"]) is not int
        or payload["format_version"] not in (1, FORMAT_VERSION)
    ):
        raise ValueError("Unsupported or incomplete training checkpoint manifest.")
    _json_bytes(payload)
    if (
        hashlib.sha256(_json_bytes(payload["identity"])).hexdigest()
        != payload["identity_sha256"]
    ):
        raise ValueError("Checkpoint configuration/data identity hash is invalid.")
    state = payload["state"]
    state_fields = {
        "global_step",
        "optimizer_steps",
        "data_cursor",
        "metrics",
        "checkpoint_path",
        "checkpoint_manifest_path",
    }
    if payload["algorithm"] == "ppo":
        state_fields |= {"critic_optimizer_steps", "critic_checkpoint_path"}
    elif payload["algorithm"] == "opd":
        state_fields.add("behavior_policy_version")
    if set(state) != state_fields or not isinstance(state["metrics"], dict):
        raise ValueError("Checkpoint trainer state is incomplete.")
    for name in ("global_step", "optimizer_steps", "data_cursor"):
        if type(state.get(name)) is not int or state[name] < 0:
            raise ValueError(f"Checkpoint {name} must be a nonnegative integer.")
    if (
        state["optimizer_steps"] < state["global_step"]
        or state["data_cursor"] < state["global_step"]
    ):
        raise ValueError("Checkpoint optimizer/data progress precedes global_step.")
    for name, component in payload["components"].items():
        if (
            name not in {"actor", "critic"}
            or not isinstance(component.get("uri"), str)
            or not component["uri"]
        ):
            raise ValueError("Checkpoint component URI is invalid.")
        if (
            type(component.get("train_tokens")) is not int
            or component["train_tokens"] < 0
        ):
            raise ValueError("Checkpoint component token accounting is invalid.")
    if payload["format_version"] == FORMAT_VERSION:
        if set(payload["schedulers"]) != set(payload["components"]):
            raise ValueError("Checkpoint scheduler components are incomplete.")
        for name, schedule in payload["schedulers"].items():
            counter = state[
                "critic_optimizer_steps" if name == "critic" else "optimizer_steps"
            ]
            if (
                set(schedule) != {"total_steps", "optimizer_steps", "last_lr"}
                or type(schedule["optimizer_steps"]) is not int
                or schedule["optimizer_steps"] != counter
            ):
                raise ValueError(
                    "Checkpoint scheduler counter does not match optimizer state."
                )
            total, rate = schedule["total_steps"], schedule["last_lr"]
            if total is not None and (type(total) is not int or total < 1):
                raise ValueError("Checkpoint scheduler horizon is invalid.")
            if (counter == 0 and rate is not None) or (
                counter > 0
                and (
                    type(rate) not in (int, float)
                    or not math.isfinite(rate)
                    or rate <= 0
                )
            ):
                raise ValueError("Checkpoint scheduler learning rate is invalid.")
    if (
        "actor" not in payload["components"]
        or "python" not in payload["rng"]
        or "numpy" not in payload["rng"]
    ):
        raise ValueError("Checkpoint actor or RNG state is missing.")
    reporting_fields = {
        "_window",
        "_second_moments",
        "_extrema",
        "_latest",
        "_last_step",
    }
    if set(payload["reporting"]) != reporting_fields or set(payload["loop"]) != {
        "kind",
        "identity",
        "state",
    }:
        raise ValueError("Checkpoint reporting or loop progress is incomplete.")
    required_runtime = {
        "sft": {"_last_eval_step", "_train_tokens"},
        "dpo": {
            "_last_eval_step",
            "_total_tokens",
            "_total_pairs",
            "_sample_input_tokens",
            "_sample_output_tokens",
        },
        "opd": {
            "_last_eval_step",
            "_total_tokens",
            "_sampling_seed_offset",
            "_sample_input_tokens",
            "_sample_output_tokens",
        },
        "ppo": {
            "_last_eval_step",
            "_next_group",
            "_sample_input_tokens",
            "_sample_output_tokens",
            "_eval_sample_input_tokens",
            "_eval_sample_output_tokens",
        },
    }
    if (
        payload["algorithm"] not in required_runtime
        or set(payload["runtime"]) != required_runtime[payload["algorithm"]]
    ):
        raise ValueError(
            "Checkpoint token, seed or evaluation counters are incomplete."
        )
    for name, value in payload["runtime"].items():
        if name == "_last_eval_step" and value is None:
            continue
        if type(value) is not int or value < 0:
            raise ValueError(
                "Checkpoint runtime counters must be nonnegative integers."
            )
    last_eval = payload["runtime"]["_last_eval_step"]
    if last_eval is not None and last_eval > state["global_step"]:
        raise ValueError("Checkpoint evaluation progress exceeds global_step.")
    if payload["components"]["actor"]["uri"] != state.get("checkpoint_path"):
        raise ValueError("Checkpoint actor URI does not match trainer state.")
    # Validate RNG payloads with isolated generators. Never mutate process RNG
    # state or create a remote run merely to discover a corrupt local record.
    import numpy as np

    random.Random().setstate(_tuple_tree(payload["rng"]["python"]))
    algorithm, keys, position, has_gauss, cached = payload["rng"]["numpy"]
    np.random.RandomState().set_state(
        (algorithm, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached)
    )
    if "torch_cpu" in payload["rng"]:
        import torch

        torch.Generator().set_state(
            torch.tensor(payload["rng"]["torch_cpu"], dtype=torch.uint8)
        )
    return payload


def training_limit(config, step, stop_after_step=None):
    """Keep the planned horizon fixed while allowing an orderly K-step pause.

    A pause skips final-only evaluation/sampling, so a resumed run observes the
    same evaluation and RNG schedule as an uninterrupted run targeting N steps.
    max_steps remains the total target, never an additional update count.
    """
    if stop_after_step is not None:
        if type(stop_after_step) is not int or stop_after_step <= step:
            raise ValueError("stop_after_step must exceed the completed global_step.")
        if config.max_steps is not None and stop_after_step > config.max_steps:
            raise ValueError("stop_after_step cannot exceed max_steps.")
    limit = stop_after_step if stop_after_step is not None else config.max_steps
    paused = stop_after_step is not None and (
        config.max_steps is None or stop_after_step < config.max_steps
    )
    return limit, paused


class TrainingCheckpoint:
    """Pair complete optimizer snapshots with one atomic local runtime record.

    Disabled by default for compatibility with streaming/remote-only recipes.
    Enabling state_dir or resume_from requires finite, stable random-access
    datasets. The manifest is published only after remote saves and local
    bookkeeping succeed. Partial updates and failed actor/critic saves leave
    the preceding manifest usable. Files are replaced atomically in one local
    directory; portability requires copying the JSON and retaining remote URIs.
    """

    def __init__(self, config, algorithm, train_dataset, eval_dataset, *, identity):
        # Resolve once from the original training plan, never from remaining
        # rows or remaining steps after a restart. This also rejects unknown
        # decay horizons before any trainer allocates a remote client.
        self.optimizer_horizons = optimizer_horizons(config, train_dataset)
        self.enabled = bool(
            config.checkpoint.state_dir or config.checkpoint.resume_from
        )
        self.config, self.algorithm = config, algorithm
        self.loaded = None
        self.started = False
        self.in_train_loop = False
        self.saved_step = None
        self.loop_owner = None
        self.loop = {"kind": "trainer", "identity": {}, "state": {}}
        self.bound = False
        if not self.enabled:
            return
        if train_dataset is None:
            raise ValueError("Complete checkpoints require a training dataset.")
        values = asdict(config)
        # Output destinations and save cadence do not affect optimization.
        # Resolved decay horizons are retained separately. Constant schedules
        # may extend the run; inferred decay horizons must remain unchanged.
        values.pop("checkpoint")
        reporting = values.pop("reporting")
        values["logging_step"] = reporting["logging_step"]
        values["training"].pop("max_steps")
        self.identity = {
            "config": values,
            "clients": identity,
            "train_data": dataset_identity(train_dataset),
            "eval_data": dataset_identity(eval_dataset),
        }
        _json_bytes(self.identity)
        if config.checkpoint.resume_from:
            self.loaded = load_training_checkpoint(config.checkpoint.resume_from)
            previous_identity = copy.deepcopy(self.loaded["identity"])
            if self.loaded["format_version"] == 1:
                # P0-02 manifests describe constant LR without a scheduler
                # module. Normalize only that historical default; changing LR
                # or introducing warmup/decay during resume remains a mismatch.
                old_config = previous_identity["config"]
                optimizers = [old_config["optimizer"]]
                if "critic" in old_config:
                    optimizers.append(old_config["critic"]["optimizer"])
                for optimizer in optimizers:
                    optimizer.setdefault(
                        "scheduler",
                        {
                            "type": "constant",
                            "warmup_steps": 0,
                            "total_steps": None,
                            "min_lr": None,
                        },
                    )
            if self.loaded["algorithm"] != algorithm or previous_identity != json.loads(
                _json_bytes(self.identity)
            ):
                raise ValueError(
                    "Checkpoint configuration, model, optimizer or dataset identity does not match."
                )
            expected = (
                {"actor", "critic"}
                if getattr(config, "use_critic", False)
                else {"actor"}
            )
            if set(self.loaded["components"]) != expected:
                raise ValueError(
                    "Checkpoint actor/critic components do not match the configuration."
                )
            expected_schedules = self._scheduler_states(self.loaded["state"])
            if self.loaded["format_version"] == 1:
                for name in expected:
                    optimizer = (
                        config.critic.optimizer
                        if name == "critic"
                        else config.optimizer
                    )
                    if (
                        optimizer.scheduler.type != "constant"
                        or optimizer.scheduler.warmup_steps
                    ):
                        raise ValueError(
                            "Version 1 checkpoints require constant LR without warmup."
                        )
            else:
                for name, schedule in expected_schedules.items():
                    previous = self.loaded["schedulers"][name]
                    lr_matches = (
                        previous["last_lr"] is schedule["last_lr"]
                        if schedule["last_lr"] is None
                        else math.isclose(
                            previous["last_lr"],
                            schedule["last_lr"],
                            rel_tol=1e-12,
                            abs_tol=1e-15,
                        )
                    )
                    if (
                        previous["total_steps"] != schedule["total_steps"]
                        or not lr_matches
                    ):
                        raise ValueError(
                            "Checkpoint scheduler horizon or LR does not match the original training plan."
                        )
            if "critic" in expected and (
                self.loaded["components"]["critic"]["uri"]
                != self.loaded["state"].get("critic_checkpoint_path")
                or self.loaded["state"].get("critic_optimizer_steps", -1) < 0
                or self.loaded["components"]["critic"]["uri"]
                == self.loaded["components"]["actor"]["uri"]
            ):
                raise ValueError(
                    "Checkpoint critic optimizer state is missing or inconsistent."
                )
            if train_dataset is None or self.loaded["state"]["data_cursor"] > len(
                train_dataset
            ):
                raise ValueError("Checkpoint data cursor exceeds the training dataset.")
            if (
                config.max_steps is not None
                and config.max_steps < self.loaded["state"]["global_step"]
            ):
                raise ValueError("max_steps cannot precede the restored global_step.")
        self.directory = (
            Path(config.checkpoint.state_dir).expanduser()
            if config.checkpoint.state_dir
            else Path(config.checkpoint.resume_from).expanduser().parent
        )

    def _scheduler_states(self, state):
        """Pair each component's schedule with its acknowledged optimizer count."""
        result = {
            "actor": scheduler_state(
                self.config.optimizer,
                state["optimizer_steps"],
                self.optimizer_horizons["actor"],
            )
        }
        if "critic" in self.optimizer_horizons:
            result["critic"] = scheduler_state(
                self.config.critic.optimizer,
                state["critic_optimizer_steps"],
                self.optimizer_horizons["critic"],
            )
        return result

    def bind_loop(self, kind="trainer", *, identity=None, owner=None):
        """Validate the application loop as well as the underlying PPO config."""
        if not self.enabled:
            return
        current = {
            "kind": kind,
            "identity": json.loads(_json_bytes(identity or {})),
            "state": {},
        }
        if self.loaded and any(
            self.loaded["loop"][key] != current[key] for key in ("kind", "identity")
        ):
            raise ValueError(
                "Checkpoint execution loop or agent configuration does not match."
            )
        self.loop, self.loop_owner, self.bound = current, owner, True
        if self.loaded and owner is not None:
            owner.restore_checkpoint_state(self.loaded["loop"]["state"])

    def create_client(self, service, **kwargs):
        if self.loaded:
            return call(
                service,
                "create_training_client_from_state_with_optimizer",
                path=self.loaded["components"]["actor"]["uri"],
            )
        return service.create_lora_training_client(**kwargs)

    async def create_client_async(self, service, *, component="actor", **kwargs):
        if self.loaded:
            return await call_async(
                service,
                "create_training_client_from_state_with_optimizer",
                path=self.loaded["components"][component]["uri"],
            )
        return await call_async(service, "create_lora_training_client", **kwargs)

    def restore(self, trainer):
        """Rehydrate local counters after algorithm/client construction completes."""
        if not self.loaded:
            return
        self.saved_step = self.loaded["state"]["global_step"]
        for key, value in self.loaded["state"].items():
            setattr(trainer.state, key, copy.deepcopy(value))
        for key, value in self.loaded["runtime"].items():
            if key in _RUNTIME_FIELDS:
                setattr(trainer, key, copy.deepcopy(value))
        actor = getattr(trainer, "_actor", None)
        if actor is not None:
            actor.optimizer_steps = trainer.state.optimizer_steps
            actor.train_tokens = self.loaded["components"]["actor"]["train_tokens"]
        critic = getattr(trainer, "critic", None)
        if "critic" in self.loaded["components"]:
            if critic is None:
                raise ValueError("Checkpoint requires a built-in trainable critic.")
            critic.optimizer_steps = trainer.state.critic_optimizer_steps
            critic.train_tokens = self.loaded["components"]["critic"]["train_tokens"]
            critic.global_step = trainer.state.global_step
            critic.checkpoint_path = trainer.state.critic_checkpoint_path
        self.reporter(trainer).load_state_dict(self.loaded["reporting"])

    @staticmethod
    def reporter(trainer):
        return (
            trainer._metrics_api
            if hasattr(trainer, "_metrics_api")
            else trainer._metrics
        )

    def begin(self):
        """Restore RNG once, immediately before the first resumed operation."""
        if not self.enabled or self.started:
            return
        if not self.bound:
            self.bind_loop()
        if self.loaded:
            restore_rng(self.loaded["rng"])
        elif self.config.seed is not None:
            import numpy as np

            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            if importlib.util.find_spec("torch") is not None:
                import torch

                torch.manual_seed(self.config.seed)
        self.started = True

    def rows(self, trainer):
        self.in_train_loop = True
        if not self.enabled:
            return iter(trainer.train_dataset)
        return training_rows(trainer.train_dataset, trainer.state)

    def require_training_loop(self):
        """Prevent manual steps from publishing an untracked dataset cursor."""
        if self.enabled and not self.in_train_loop:
            raise ValueError(
                "Complete checkpoint updates require train(); manual step() has no resumable data cursor."
            )

    def end(self):
        self.in_train_loop = False

    def evaluation_due(self, trainer, last_eval_step):
        """Replay an evaluation interrupted after a saved optimizer boundary."""
        config, step = trainer.args, trainer.state.global_step
        return trainer.eval_dataset is not None and (
            (config.eval_before_train and last_eval_step is None)
            or (
                self.enabled
                and step > 0
                and config.eval_steps > 0
                and step % config.eval_steps == 0
                and last_eval_step != step
            )
        )

    def remote_saved(self, trainer, *, publish=True):
        """Mark the exact remote boundary eligible for local manifest refresh."""
        self.saved_step = trainer.state.global_step
        # AgentTrainingLoop still needs to commit candidate history and sample
        # accounting after PPOTrainer.step returns. Its update/finalization
        # boundary publishes the paired record once that bookkeeping is done.
        if not publish or self.loop_owner is not None:
            return None
        return self.publish(trainer)

    def publish(self, trainer):
        """Publish only a fully acknowledged actor/critic update boundary."""
        if not self.enabled:
            return None
        if self.saved_step != trainer.state.global_step:
            return None
        if not self.bound:
            self.bind_loop()
        if not trainer.state.checkpoint_path:
            raise ValueError("Cannot publish runtime state without saved actor state.")
        state = asdict(trainer.state)
        self.directory.mkdir(parents=True, exist_ok=True)
        target = (
            self.directory
            / f"{self.algorithm}-step-{trainer.state.global_step:08d}.json"
        )
        state["checkpoint_manifest_path"] = str(target.resolve())
        actor = getattr(trainer, "_actor", None)
        components = {
            "actor": {
                "uri": trainer.state.checkpoint_path,
                "train_tokens": actor.train_tokens
                if actor
                else getattr(
                    trainer, "_train_tokens", getattr(trainer, "_total_tokens", 0)
                ),
            }
        }
        if getattr(trainer.args, "use_critic", False):
            if not trainer.state.critic_checkpoint_path:
                raise ValueError(
                    "Cannot publish a PPO checkpoint without critic state."
                )
            components["critic"] = {
                "uri": trainer.state.critic_checkpoint_path,
                "train_tokens": trainer.critic.train_tokens,
            }
        loop = copy.deepcopy(self.loop)
        if self.loop_owner is not None:
            loop["state"] = self.loop_owner.checkpoint_state()
        payload = {
            "format_version": FORMAT_VERSION,
            "algorithm": self.algorithm,
            "identity": self.identity,
            "identity_sha256": hashlib.sha256(_json_bytes(self.identity)).hexdigest(),
            "components": components,
            "schedulers": self._scheduler_states(state),
            "state": state,
            "runtime": {
                key: getattr(trainer, key)
                for key in _RUNTIME_FIELDS
                if hasattr(trainer, key)
            },
            "rng": capture_rng(),
            "loop": loop,
            "reporting": self.reporter(trainer).state_dict(),
        }
        encoded = _json_bytes(payload)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory, prefix=".state-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(encoded + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        trainer.state.checkpoint_manifest_path = str(target.resolve())
        return str(target.resolve())
