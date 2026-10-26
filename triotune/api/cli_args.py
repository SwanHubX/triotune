"""Composable configuration contracts for the PyTRIO trainers.

The configuration hierarchy is organized by runtime responsibility rather
than by inheritance depth. Every top-level algorithm configuration composes
small modules for the training loop, optimizer, checkpoints, evaluation and
reporting, then adds only the modules needed by that algorithm. PyTRIO owns
model placement, distributed execution and inference infrastructure, so this
module intentionally does not expose vLLM, FSDP, Megatron or cluster settings.

New code should use nested modules, for example::

    PPOConfig(
        training=TrainingConfig(batch_size=4, max_steps=100),
        optimizer=OptimizerConfig(learning_rate=4e-5),
        rollout=PPORolloutConfig(num_generations=4, max_tokens=1024),
        actor=PPOActorConfig(eps_clip=0.2),
        critic=PPOCriticConfig(enabled=True),
    )

The former flat keyword API remains accepted during migration. Flat attribute
reads and assignments are routed to their owning module, so existing trainers
and callers continue to work while serialized configurations become explicitly
modular. The compatibility layer is deliberately kept at this public boundary;
algorithm implementations do not need duplicate legacy branches.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal


# ---------------------------------------------------------------------------
# Shared validation and module construction helpers
# ---------------------------------------------------------------------------


def _integer(name: str, value: Any, *, minimum: int) -> None:
    """Require an integer at or above an inclusive lower bound."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _positive(name: str, value: Any) -> None:
    """Require a finite, strictly positive number."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive.")


def _nonnegative(name: str, value: Any) -> None:
    """Require a finite number greater than or equal to zero."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative.")


def _probability(name: str, value: Any, *, include_zero: bool) -> None:
    """Validate a coefficient whose upper bound is one."""

    valid = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (0 <= value <= 1 if include_zero else 0 < value <= 1)
    )
    if not valid:
        interval = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"{name} must be finite and in {interval}.")


def _non_empty_string(name: str, value: Any) -> None:
    """Require non-whitespace text for an externally visible identifier."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _chat_template_options(options: Any, reserved: set[str]) -> None:
    """Reject tokenizer arguments owned by the dataset preparation layer."""

    if not isinstance(options, dict):
        raise ValueError("chat_template_kwargs must be a dictionary.")
    conflicts = reserved.intersection(options)
    if conflicts:
        raise ValueError(
            "chat_template_kwargs must not override "
            + ", ".join(sorted(conflicts))
            + "."
        )


def _coerce_module(
    value: Any,
    module_type: type,
    *,
    defaults: Mapping[str, Any] | None = None,
):
    """Build one typed module from an instance, mapping or omitted value."""

    if isinstance(value, module_type):
        return value
    values = dict(defaults or {})
    if value is not None:
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{module_type.__name__} must be supplied as an instance or mapping."
            )
        values.update(value)
    return module_type(**values)


def _coerce_optimizer(
    value: Any, *, learning_rate: float = 1e-5
) -> "OptimizerConfig":
    """Coerce optimizer mappings while accepting the common ``lr`` alias."""

    if isinstance(value, OptimizerConfig):
        return value
    values: dict[str, Any] = {"learning_rate": learning_rate}
    if value is not None:
        if not isinstance(value, Mapping):
            raise TypeError("optimizer must be an OptimizerConfig or mapping.")
        values.update(value)
    if "lr" in values:
        values["learning_rate"] = values.pop("lr")
    return OptimizerConfig(**values)


def _module_values(
    module_type: type,
    source: Mapping[str, Any] | None,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Select fields owned by one module from a larger YAML section."""

    if not isinstance(source, Mapping):
        return {}
    names = {item.name for item in fields(module_type)}
    aliases = aliases or {}
    result = {}
    for key, value in source.items():
        target = aliases.get(key, key)
        if target in names:
            result[target] = value
    return result


def _reference_values(source: Any) -> Any:
    """Normalize the historical ``reference.path`` spelling to ``name``."""

    if not isinstance(source, Mapping):
        return source
    values = _module_values(ReferenceConfig, source)
    if "name" not in values and "path" in source:
        values["name"] = source["path"]
    return values


# ---------------------------------------------------------------------------
# Framework-wide functional modules
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelConfig:
    """Describe the trainable PyTRIO model owned by an experiment.

    Parameters:
        name (`str` or `None`, default `None`):
            PyTRIO base-model identifier passed to
            ``create_lora_training_client``. ``None`` keeps the historical API
            usable when the model is still supplied directly to a trainer;
            complete YAML experiments should always set this field.
        lora_rank (`int`, default `32`):
            Rank of the trainable LoRA adapter. Larger ranks increase adapter
            capacity and remote training cost. The value must be positive.

    Model placement, numerical precision, sharding and device topology are not
    represented here because the PyTRIO service owns those decisions.
    """

    name: str | None = None
    lora_rank: int = 32

    def __post_init__(self) -> None:
        if self.name is not None:
            _non_empty_string("model.name", self.name)
        _integer("model.lora_rank", self.lora_rank, minimum=1)


@dataclass(slots=True)
class DatasetConfig:
    """Describe one raw dataset source before algorithm-specific preparation.

    Parameters:
        path (`str` or `None`, default `None`):
            Local file path or dataset repository identifier. ``None`` supports
            callers that inject an in-memory iterable directly into a trainer;
            complete YAML experiments should provide a path.
        split (`str`, default `"train"`):
            Logical source split requested by the recipe-side loader. Triotune
            does not impose Hugging Face semantics; a JSONL loader may simply
            treat this as descriptive metadata.
        max_examples (`int` or `None`, default `None`):
            Optional positive cap applied after loading. ``None`` preserves the
            complete source. This is useful for smoke tests without changing
            optimizer step semantics.
        shuffle (`bool`, default `False`):
            Whether the recipe should randomize source order before iteration.
            Trainers themselves consume the iterable they receive and do not
            silently reshuffle it.
        seed (`int` or `None`, default `None`):
            Dataset-only shuffle seed. ``None`` lets a recipe reuse
            ``training.seed`` or retain source order.
        loader_options (`dict`, default `{}`):
            Loader-specific, non-secret options such as a dataset subset or
            revision. Keeping them in one mapping avoids adding backend fields
            to the core contract while still permitting custom data adapters.
    """

    path: str | None = None
    split: str = "train"
    max_examples: int | None = None
    shuffle: bool = False
    seed: int | None = None
    loader_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.path is not None:
            _non_empty_string("dataset.path", self.path)
        _non_empty_string("dataset.split", self.split)
        if self.max_examples is not None:
            _integer("dataset.max_examples", self.max_examples, minimum=1)
        if not isinstance(self.shuffle, bool):
            raise ValueError("dataset.shuffle must be boolean.")
        if self.seed is not None:
            _integer("dataset.seed", self.seed, minimum=0)
        if not isinstance(self.loader_options, dict):
            raise ValueError("dataset.loader_options must be a dictionary.")


@dataclass(slots=True)
class ReferenceConfig:
    """Describe an optional frozen reference-policy base model.

    Parameters:
        name (`str` or `None`, default `None`):
            PyTRIO base-model identifier used for fixed log-probability scoring.
            ``None`` means "reuse the trainable model's unmodified base model",
            which is the normal PPO and DPO default.

    A reference has no optimizer or LoRA rank because Triotune never updates it.
    """

    name: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            _non_empty_string("reference.name", self.name)


@dataclass(slots=True)
class TeacherConfig:
    """Describe the fixed teacher used by a single-teacher OPD experiment.

    Parameters:
        name (`str` or `None`, default `None`):
            Teacher base-model identifier. ``None`` preserves compatibility
            with callers that still pass ``teacher_model`` to ``OPDTrainer``;
            complete OPD YAML should provide a value.
        model_path (`str` or `None`, default `None`):
            Optional PyTRIO checkpoint URI loaded on top of ``name``. ``None``
            scores with the untouched teacher base model.

    Multi-teacher routing remains an explicit ``OPDTeacherSpec`` runtime API;
    this compact module covers the common one-teacher experiment contract.
    """

    name: str | None = None
    model_path: str | None = None

    def __post_init__(self) -> None:
        if self.name is not None:
            _non_empty_string("teacher.name", self.name)
        if self.model_path is not None:
            _non_empty_string("teacher.model_path", self.model_path)


@dataclass(slots=True)
class TrainingConfig:
    """Control dataset consumption and the trainer's update loop.

    Parameters:
        batch_size (`int`, default `1`):
            Number of algorithm-level examples consumed by one trainer update:
            prompts for PPO/OPD, preference pairs for DPO, and supervised
            sequences for SFT. It is not a device microbatch because PyTRIO
            owns remote execution.
        max_steps (`int` or `None`, default `None`):
            Maximum completed trainer updates. ``None`` consumes the supplied
            training iterable once, so this is a safety ceiling rather than an
            instruction to cycle exhausted data.
        seed (`int` or `None`, default `None`):
            Nonnegative seed forwarded to PyTRIO client creation and used as
            the base for per-request sampling seeds. ``None`` delegates seed
            selection to the service.
    """

    batch_size: int = 1
    max_steps: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        _integer("batch_size", self.batch_size, minimum=1)
        if self.max_steps is not None:
            _integer("max_steps", self.max_steps, minimum=1)
        if self.seed is not None:
            _integer("seed", self.seed, minimum=0)


@dataclass(slots=True)
class OptimizerConfig:
    """Describe the AdamW parameters accepted by PyTRIO.

    Parameters:
        learning_rate (`float`, default `1e-5`):
            AdamW step size. The conservative default is shared by PPO/OPD;
            DPO and SFT top-level configs override it with task-specific values.
        beta1 (`float`, default `0.9`):
            Exponential decay for the first moment. Lower values react faster
            to changing gradients but produce noisier momentum.
        beta2 (`float`, default `0.95`):
            Exponential decay for the squared-gradient moment. ``0.95`` adapts
            faster than the conventional ``0.999`` in short fine-tuning runs.
        eps (`float`, default `1e-8`):
            Positive denominator stabilizer used by AdamW.
        weight_decay (`float`, default `0.0`):
            Decoupled parameter shrinkage. Zero disables weight decay, which is
            the safe default for small LoRA adapters.
        grad_clip_norm (`float`, default `0.0`):
            Global gradient-norm ceiling applied remotely. Zero disables
            clipping; a positive value limits unstable updates.

    PyTRIO creates and owns the optimizer. Scheduler, precision, sharding and
    distributed optimizer settings are therefore intentionally absent.
    """

    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0

    def __post_init__(self) -> None:
        _positive("learning_rate", self.learning_rate)
        _positive("eps", self.eps)
        for name in ("beta1", "beta2"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value < 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1).")
        _nonnegative("weight_decay", self.weight_decay)
        _nonnegative("grad_clip_norm", self.grad_clip_norm)

    def optimizer_params(self) -> dict[str, float]:
        """Return the SDK-neutral mapping consumed by the PyTRIO adapter."""

        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
        }


@dataclass(slots=True)
class CheckpointConfig:
    """Configure periodic checkpoint creation.

    Parameters:
        save_every (`int`, default `0`):
            Interval measured in completed trainer updates. Zero disables
            periodic saves; trainers still create their documented final
            checkpoint so a successful run always exposes its result.
    """

    save_every: int = 0

    def __post_init__(self) -> None:
        _integer("save_every", self.save_every, minimum=0)


@dataclass(slots=True)
class EvaluationConfig:
    """Configure evaluation timing and bounded dataset consumption.

    Parameters:
        steps (`int`, default `0`):
            Run evaluation after every N completed updates. Zero disables the
            periodic trigger.
        before_train (`bool`, default `False`):
            Evaluate the initial policy before any optimizer update.
        after_train (`bool`, default `True`):
            Evaluate the final policy unless the same step was already
            evaluated. Individual algorithm configs may override this default
            when final evaluation would add expensive generation requests.
        batch_size (`int` or `None`, default `None`):
            Rows per evaluation batch. ``None`` reuses training batch size.
        max_batches (`int` or `None`, default `None`):
            Optional positive limit for one evaluation pass. ``None`` consumes
            the full validation iterable.

    Evaluation data is described by ``valid_dataset`` and materialized by the
    caller; this module controls only scheduling and bounded consumption.
    """

    steps: int = 0
    before_train: bool = False
    after_train: bool = True
    batch_size: int | None = None
    max_batches: int | None = None

    def __post_init__(self) -> None:
        _integer("evaluation.steps", self.steps, minimum=0)
        for name in ("before_train", "after_train"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"evaluation.{name} must be boolean.")
        for name in ("batch_size", "max_batches"):
            value = getattr(self, name)
            if value is not None:
                _integer(f"evaluation.{name}", value, minimum=1)


@dataclass(slots=True)
class ReportingConfig:
    """Configure metric aggregation and optional SwanLab reporting.

    Parameters:
        report_to (`"swanlab"` or `None`, default `None`):
            Optional reporting backend. ``None`` avoids network reporting and
            keeps terminal progress only.
        project (`str`, default `"trio-cookbook"`):
            SwanLab project receiving the run. Algorithm configs provide more
            specific defaults where useful.
        workspace (`str` or `None`, default `None`):
            SwanLab workspace or organization. ``None`` uses the account default.
        tags (`list[str]` or `None`, default `None`):
            Optional nonempty labels used to search and compare runs.
        run_name (`str` or `None`, default `None`):
            Explicit display name. ``None`` lets the trainer derive one from
            the model and algorithm.
        log_dir (`str` or `None`, default `None`):
            Optional local output directory. Trainers append a unique JSONL
            history containing step metrics, checkpoint URIs and final status;
            SwanLab also stores its local artifacts here when enabled.
        logging_step (`int`, default `1`):
            Number of updates accumulated before DPO/OPD metric windows flush.
            Trainers that publish every step ignore this cadence.
    """

    report_to: Literal["swanlab"] | None = None
    project: str = "trio-cookbook"
    workspace: str | None = None
    tags: list[str] | None = None
    run_name: str | None = None
    log_dir: str | None = None
    logging_step: int = 1

    def __post_init__(self) -> None:
        if self.report_to not in (None, "swanlab"):
            raise ValueError("report_to must be 'swanlab' or None.")
        _non_empty_string("project", self.project)
        for name in ("workspace", "run_name", "log_dir"):
            value = getattr(self, name)
            if value is not None:
                _non_empty_string(name, value)
        if self.tags is not None and (
            not isinstance(self.tags, list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in self.tags)
        ):
            raise ValueError("tags must be a list of non-empty strings when provided.")
        _integer("logging_step", self.logging_step, minimum=1)


@dataclass(frozen=True, slots=True)
class NormConfig:
    """Select the populations used to center and scale PPO values.

    Parameters:
        mean_level (`"batch"`, `"group"` or `None`, default `"group"`):
            Population used to subtract a mean. Group statistics compare
            responses to the same prompt; ``None`` preserves the raw location.
        std_level (`"batch"`, `"group"` or `None`, default `"group"`):
            Population used for standard-deviation scaling. ``None`` performs
            centering without variance normalization.
        mean_leave1out (`bool`, default `False`):
            Exclude the current response from its group mean. This produces the
            leave-one-out baseline used by RLOO.
        std_unbiased (`bool`, default `True`):
            Use the sample standard deviation when at least two values exist.
        eps (`float`, default `1e-5`):
            Positive stabilizer added to the denominator after scaling.
    """

    mean_level: Literal["batch", "group"] | None = "group"
    std_level: Literal["batch", "group"] | None = "group"
    mean_leave1out: bool = False
    std_unbiased: bool = True
    eps: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("mean_leave1out", "std_unbiased"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        for level in (self.mean_level, self.std_level):
            if level not in (None, "batch", "group"):
                raise ValueError("Normalization levels must be batch, group or None.")
        _positive("normalization eps", self.eps)


# ---------------------------------------------------------------------------
# Common top-level composition and flat-API compatibility
# ---------------------------------------------------------------------------


@dataclass(init=False)
class BaseTrainerConfig:
    """Compose the experiment modules shared by every trainer.

    This class intentionally owns no algorithm fields. Derived configurations
    add reference, teacher, rollout, data, objective or execution modules by
    composition. ``model`` and dataset sources make each derived configuration
    capable of representing a complete YAML experiment, while their optional
    defaults preserve callers that inject those dependencies directly.

    Parameters:
        model (`ModelConfig` or `None`, default `None`):
            Trainable model identity and LoRA rank. Complete experiments set it;
            legacy trainer construction may continue to pass the model directly.
        train_dataset (`DatasetConfig` or `None`, default `None`):
            Raw training source description. The trainer receives a materialized
            iterable, so recipe code remains responsible for opening the source.
        valid_dataset (`DatasetConfig` or `None`, default `None`):
            Optional validation source. ``None`` disables dataset-backed
            evaluation independently of the evaluation schedule.
        training (`TrainingConfig`): Batch, step and reproducibility controls.
        optimizer (`OptimizerConfig`): AdamW parameters for the trainable model.
        checkpoint (`CheckpointConfig`): Periodic remote checkpoint cadence.
        evaluation (`EvaluationConfig`): Evaluation schedule and batch limits.
        reporting (`ReportingConfig`): Metric backend and experiment metadata.

    Flat attribute routes are a migration aid and are not serialized by
    ``dataclasses.asdict``.
    """

    model: ModelConfig | None
    train_dataset: DatasetConfig | None
    valid_dataset: DatasetConfig | None
    training: TrainingConfig
    optimizer: OptimizerConfig
    checkpoint: CheckpointConfig
    evaluation: EvaluationConfig
    reporting: ReportingConfig

    _COMMON_ROUTES: ClassVar[dict[str, tuple[str, ...]]] = {
        "batch_size": ("training", "batch_size"),
        "max_steps": ("training", "max_steps"),
        "seed": ("training", "seed"),
        "learning_rate": ("optimizer", "learning_rate"),
        "beta1": ("optimizer", "beta1"),
        "beta2": ("optimizer", "beta2"),
        "eps": ("optimizer", "eps"),
        "weight_decay": ("optimizer", "weight_decay"),
        "grad_clip_norm": ("optimizer", "grad_clip_norm"),
        "save_every": ("checkpoint", "save_every"),
        "eval_steps": ("evaluation", "steps"),
        "eval_before_train": ("evaluation", "before_train"),
        "eval_after_train": ("evaluation", "after_train"),
        "eval_batch_size": ("evaluation", "batch_size"),
        "eval_max_batches": ("evaluation", "max_batches"),
        "report_to": ("reporting", "report_to"),
        "project": ("reporting", "project"),
        "workspace": ("reporting", "workspace"),
        "tags": ("reporting", "tags"),
        "run_name": ("reporting", "run_name"),
        "log_dir": ("reporting", "log_dir"),
        "logging_step": ("reporting", "logging_step"),
    }

    def __init__(
        self,
        *,
        model: ModelConfig | Mapping[str, Any] | None = None,
        train_dataset: DatasetConfig | Mapping[str, Any] | None = None,
        valid_dataset: DatasetConfig | Mapping[str, Any] | None = None,
        training: TrainingConfig | Mapping[str, Any] | None = None,
        optimizer: OptimizerConfig | Mapping[str, Any] | None = None,
        checkpoint: CheckpointConfig | Mapping[str, Any] | None = None,
        evaluation: EvaluationConfig | Mapping[str, Any] | None = None,
        reporting: ReportingConfig | Mapping[str, Any] | None = None,
        **legacy: Any,
    ) -> None:
        self._init_common(
            training,
            optimizer,
            checkpoint,
            evaluation,
            reporting,
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        )
        self._apply_legacy(legacy)
        self.__post_init__()

    def _init_common(
        self,
        training: Any,
        optimizer: Any,
        checkpoint: Any,
        evaluation: Any,
        reporting: Any,
        *,
        model: Any = None,
        train_dataset: Any = None,
        valid_dataset: Any = None,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        project: str = "trio-cookbook",
        eval_after_train: bool = True,
    ) -> None:
        """Construct common modules with algorithm-specific default overrides."""

        object.__setattr__(
            self,
            "model",
            None if model is None else _coerce_module(model, ModelConfig),
        )
        object.__setattr__(
            self,
            "train_dataset",
            None
            if train_dataset is None
            else _coerce_module(train_dataset, DatasetConfig),
        )
        object.__setattr__(
            self,
            "valid_dataset",
            None
            if valid_dataset is None
            else _coerce_module(valid_dataset, DatasetConfig),
        )
        object.__setattr__(
            self,
            "training",
            _coerce_module(training, TrainingConfig, defaults={"batch_size": batch_size}),
        )
        object.__setattr__(
            self,
            "optimizer",
            _coerce_optimizer(optimizer, learning_rate=learning_rate),
        )
        object.__setattr__(self, "checkpoint", _coerce_module(checkpoint, CheckpointConfig))
        object.__setattr__(
            self,
            "evaluation",
            _coerce_module(
                evaluation,
                EvaluationConfig,
                defaults={"after_train": eval_after_train},
            ),
        )
        object.__setattr__(
            self,
            "reporting",
            _coerce_module(reporting, ReportingConfig, defaults={"project": project}),
        )

    @classmethod
    def _flat_routes(cls) -> dict[str, tuple[str, ...]]:
        return cls._COMMON_ROUTES

    def __getattr__(self, name: str) -> Any:
        route = type(self)._flat_routes().get(name)
        if route is None:
            raise AttributeError(name)
        value: Any = object.__getattribute__(self, route[0])
        for part in route[1:]:
            value = getattr(value, part)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        route = type(self)._flat_routes().get(name)
        if route is None or route[0] not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        owner: Any = object.__getattribute__(self, route[0])
        for part in route[1:-1]:
            owner = getattr(owner, part)
        setattr(owner, route[-1], value)

    def _apply_legacy(self, values: Mapping[str, Any]) -> None:
        """Route former flat constructor keywords into their owning modules."""

        routes = type(self)._flat_routes()
        unknown = sorted(set(values).difference(routes))
        if unknown:
            joined = ", ".join(unknown)
            raise TypeError(f"Unexpected configuration field(s): {joined}.")
        for name, value in values.items():
            # The old flat API required callers to construct NormConfig
            # explicitly. Mapping coercion belongs to the new actor module and
            # YAML loaders, not to legacy scalar-field compatibility.
            if name in {"reward_norm", "adv_norm"} and isinstance(value, Mapping):
                raise ValueError(f"{name} must be a NormConfig or None.")
            setattr(self, name, value)

    def __post_init__(self) -> None:
        """Verify that every shared responsibility is represented by its module."""

        expected = {
            "training": TrainingConfig,
            "optimizer": OptimizerConfig,
            "checkpoint": CheckpointConfig,
            "evaluation": EvaluationConfig,
            "reporting": ReportingConfig,
        }
        for name, module_type in expected.items():
            module = getattr(self, name)
            if not isinstance(module, module_type):
                raise TypeError(f"{name} must be a {module_type.__name__}.")
            # Legacy flat keywords are routed after module construction. Run
            # validation again so compatibility never bypasses module invariants.
            module.__post_init__()
        optional = {
            "model": ModelConfig,
            "train_dataset": DatasetConfig,
            "valid_dataset": DatasetConfig,
        }
        for name, module_type in optional.items():
            module = getattr(self, name)
            if module is not None:
                if not isinstance(module, module_type):
                    raise TypeError(f"{name} must be a {module_type.__name__} or None.")
                module.__post_init__()

    def optimizer_params(self) -> dict[str, float]:
        """Preserve the historical trainer-level optimizer adapter."""

        return self.optimizer.optimizer_params()


def _common_modules(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Extract common modules from a modular YAML mapping.

    Older recipe files placed checkpoint and evaluation keys inside
    ``training``. They remain accepted, while new files should use dedicated
    ``checkpoint`` and ``evaluation`` sections.
    """

    training_source = settings.get("training")
    training = _module_values(
        TrainingConfig,
        training_source,
        {"steps": "max_steps", "total_train_steps": "max_steps"},
    )
    optimizer = _module_values(
        OptimizerConfig, settings.get("optimizer"), {"lr": "learning_rate"}
    )
    checkpoint = _module_values(CheckpointConfig, settings.get("checkpoint"))
    evaluation = _module_values(EvaluationConfig, settings.get("evaluation"))
    reporting = _module_values(ReportingConfig, settings.get("reporting"))

    if isinstance(training_source, Mapping):
        checkpoint.update(_module_values(CheckpointConfig, training_source))
        evaluation.update(
            {
                target: training_source[source]
                for source, target in {
                    "eval_steps": "steps",
                    "eval_before_train": "before_train",
                    "eval_after_train": "after_train",
                    "eval_batch_size": "batch_size",
                    "eval_max_batches": "max_batches",
                }.items()
                if source in training_source
            }
        )

    for section in ("logging", "swanlab"):
        reporting.update(
            _module_values(
                ReportingConfig,
                settings.get(section),
                {"name": "run_name"},
            )
        )
    return {
        "model": settings.get("model"),
        "train_dataset": settings.get("train_dataset"),
        "valid_dataset": settings.get("valid_dataset"),
        "training": training,
        "optimizer": optimizer,
        "checkpoint": checkpoint,
        "evaluation": evaluation,
        "reporting": reporting,
    }


# ---------------------------------------------------------------------------
# PPO-family modules
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PPORolloutConfig:
    """Configure prompt rendering and on-policy completion sampling.

    Parameters:
        num_generations (`int`, default `4`):
            Responses sampled for every prompt. Multiple responses are needed
            by group-relative algorithms and improve PPO exploration at extra
            sampling cost.
        max_tokens (`int`, default `512`):
            Maximum completion tokens sent to ``SamplingParams``. This excludes
            prompt tokens.
        temperature (`float`, default `1.0`):
            Logit temperature used by sampling. The current PPO target-logprob
            contract requires exactly ``1.0``.
        top_p (`float`, default `1.0`):
            Nucleus probability mass. The current PPO contract requires ``1.0``
            because target-only logprobs cannot reconstruct truncated support.
        prompt_key (`str`, default `"messages"`):
            Raw dataset field containing text, token IDs or chat messages.
        chat_template_kwargs (`dict`, default `{}`):
            Model-specific options passed to tokenizer chat rendering. Core
            tokenization and generation-prefix controls remain trainer-owned.

    PyTRIO supplies the sampling service, so engine topology and serving
    backend settings do not belong in this request-level module.
    """

    num_generations: int = 4
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    prompt_key: str = "messages"
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _integer("num_generations", self.num_generations, minimum=1)
        _integer("max_tokens", self.max_tokens, minimum=1)
        _positive("temperature", self.temperature)
        _probability("top_p", self.top_p, include_zero=False)
        _non_empty_string("prompt_key", self.prompt_key)
        _chat_template_options(
            self.chat_template_kwargs, {"tokenize", "add_generation_prompt"}
        )


@dataclass(slots=True)
class PPOActorConfig:
    """Configure reward processing, advantage estimation and policy updates.

    Parameters:
        reward_norm (`NormConfig` or `None`, default `None`):
            Optional normalization of scalar outcome rewards before assigning
            them to completion tokens. Standard PPO uses the critic baseline
            and leaves this disabled; GRPO commonly uses group normalization.
        adv_norm (`NormConfig` or `None`, default batch normalization):
            Optional normalization after GAE. Batch centering/scaling stabilizes
            policy updates without changing within-trajectory credit assignment.
        reward_bias (`float`, default `0.0`):
            Additive reward shift applied before scaling.
        reward_scaling (`float`, default `1.0`):
            Positive multiplier applied after ``reward_bias``.
        reward_clip (`float`, default `20.0`):
            Positive symmetric bound applied to shaped outcome rewards.
        discount (`float`, default `1.0`):
            GAE reward discount. One preserves undiscounted final-answer reward.
        gae_lambda (`float`, default `1.0`):
            GAE trace coefficient. One uses full Monte Carlo-style propagation.
        gae_timestep_unit (`"token"` or `"turn"`, default `"token"`):
            Unit over which discount and lambda advance.
        kl_ctl (`float`, default `0.0`):
            Nonnegative coefficient for fixed-reference KL reward. Zero skips
            reference scoring entirely.
        kl_estimator (`"k1"`, `"k2"` or `"k3"`, default `"k1"`):
            Token-level estimator used to convert policy/reference log-ratio
            differences into a KL penalty.
        mask_no_eos_with_zero (`bool`, default `False`):
            Replace task reward with zero for length-truncated responses while
            retaining KL reward and critic bootstrap behavior.
        eps_clip (`float`, default `0.2`):
            Lower and default upper PPO ratio-clipping width.
        eps_clip_higher (`float` or `None`, default `None`):
            Optional asymmetric upper clipping width. ``None`` reuses
            ``eps_clip``.
        c_clip (`float` or `None`, default `None`):
            Optional dual-clip coefficient greater than one for negative
            advantages. ``None`` selects ordinary PPO clipping.
        loss_type (`"ppo"`, `"sapo"` or `"cispo"`, default `"ppo"`):
            Policy surrogate evaluated by the custom loss bridge.
        importance_sampling_level (`"token"` or `"sequence"`, default `"token"`):
            Granularity of the policy ratio. Sequence mode produces the GSPO
            geometric ratio.
        loss_aggregation (`str`, default `"token_mean"`):
            Reduction denominator: active tokens, sequences, prompt groups, or
            a configured constant.
        loss_aggregation_divisor (`float` or `None`, default `None`):
            Positive fixed denominator required only by constant aggregation.
        use_decoupled_loss (`bool`, default `False`):
            Separate rollout behavior logprobs from the proximal denominator.
        recompute_logprob (`bool`, default `False`):
            Re-score rollouts with the current actor before freezing PPO targets.
        rejection_lower (`float` or `None`, default `None`):
            Optional lower behavior/proximal-ratio rejection threshold.
        rejection_upper (`float` or `None`, default `None`):
            Optional upper behavior/proximal-ratio rejection threshold.
        sapo_tau_pos (`float`, default `1.0`):
            SAPO temperature for positive-advantage tokens.
        sapo_tau_neg (`float`, default `1.05`):
            SAPO temperature for non-positive-advantage tokens.
        ppo_n_minibatches (`int`, default `1`):
            Actor optimizer partitions per frozen rollout batch. Prompt groups
            remain intact.
        update_epochs (`int`, default `1`):
            Passes over the same frozen actor targets.
        loss_backend (`"custom"` or `"native"`, default `"custom"`):
            Use Triotune's full loss implementation or PyTRIO's restricted
            built-in PPO loss.
        dynamic_sampling (`bool`, default `False`):
            Reject equal-reward prompt groups and sample replacements.
        max_empty_batches (`int`, default `32`):
            Consecutive uninformative refill limit before dynamic sampling fails.
        overlong_tokens (`int`, default `0`):
            Completion budget where the optional length penalty begins. Zero
            disables that penalty.
        overlong_penalty_factor (`float`, default `1.0`):
            Maximum positive scale of the overlong completion penalty.

    Adam parameters live in the top-level ``optimizer`` module and generation
    requests live in ``rollout`` so policy behavior is not mixed with runtime
    ownership.
    """

    reward_norm: NormConfig | None = None
    adv_norm: NormConfig | None = field(
        default_factory=lambda: NormConfig("batch", "batch")
    )
    reward_bias: float = 0.0
    reward_scaling: float = 1.0
    reward_clip: float = 20.0
    discount: float = 1.0
    gae_lambda: float = 1.0
    gae_timestep_unit: Literal["token", "turn"] = "token"
    kl_ctl: float = 0.0
    kl_estimator: Literal["k1", "k2", "k3"] = "k1"
    mask_no_eos_with_zero: bool = False
    eps_clip: float = 0.2
    eps_clip_higher: float | None = None
    c_clip: float | None = None
    loss_type: Literal["ppo", "sapo", "cispo"] = "ppo"
    importance_sampling_level: Literal["token", "sequence"] = "token"
    loss_aggregation: Literal[
        "token_mean", "seq_mean", "prompt_mean", "constant"
    ] = "token_mean"
    loss_aggregation_divisor: float | None = None
    use_decoupled_loss: bool = False
    recompute_logprob: bool = False
    rejection_lower: float | None = None
    rejection_upper: float | None = None
    sapo_tau_pos: float = 1.0
    sapo_tau_neg: float = 1.05
    ppo_n_minibatches: int = 1
    update_epochs: int = 1
    loss_backend: Literal["custom", "native"] = "custom"
    dynamic_sampling: bool = False
    max_empty_batches: int = 32
    overlong_tokens: int = 0
    overlong_penalty_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in ("reward_norm", "adv_norm"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                value = NormConfig(**dict(value))
                setattr(self, name, value)
            if value is not None and not isinstance(value, NormConfig):
                raise ValueError(f"{name} must be a NormConfig or None.")
        for name in (
            "mask_no_eos_with_zero",
            "use_decoupled_loss",
            "recompute_logprob",
            "dynamic_sampling",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        for name in ("ppo_n_minibatches", "update_epochs", "max_empty_batches"):
            _integer(name, getattr(self, name), minimum=1)
        _integer("overlong_tokens", self.overlong_tokens, minimum=0)
        for name in (
            "reward_scaling",
            "reward_clip",
            "sapo_tau_pos",
            "sapo_tau_neg",
        ):
            _positive(name, getattr(self, name))
        for name in ("eps_clip", "kl_ctl", "overlong_penalty_factor"):
            _nonnegative(name, getattr(self, name))
        for name in ("discount", "gae_lambda"):
            _probability(name, getattr(self, name), include_zero=True)
        if not math.isfinite(self.reward_bias) or self.eps_clip > 1:
            raise ValueError("reward_bias must be finite and eps_clip <= 1.")
        for name in ("eps_clip_higher", "rejection_lower", "rejection_upper"):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        if self.c_clip is not None and (
            not math.isfinite(self.c_clip) or self.c_clip <= 1
        ):
            raise ValueError("c_clip must be finite and > 1.")
        choices = {
            "gae_timestep_unit": ("token", "turn"),
            "kl_estimator": ("k1", "k2", "k3"),
            "loss_type": ("ppo", "sapo", "cispo"),
            "importance_sampling_level": ("token", "sequence"),
            "loss_aggregation": (
                "token_mean",
                "seq_mean",
                "prompt_mean",
                "constant",
            ),
            "loss_backend": ("custom", "native"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}.")
        if self.loss_aggregation == "constant":
            _positive("loss_aggregation_divisor", self.loss_aggregation_divisor)
        elif self.loss_aggregation_divisor is not None:
            raise ValueError(
                "loss_aggregation_divisor only applies to constant reduction."
            )
        if self.loss_type == "sapo" and self.use_decoupled_loss:
            raise ValueError("SAPO requires use_decoupled_loss=False.")
        if self.loss_type == "cispo" and (
            self.importance_sampling_level != "token"
            or self.eps_clip_higher is None
        ):
            raise ValueError("CISPO requires token ratios and eps_clip_higher.")
        if (
            self.rejection_lower is not None
            and self.rejection_upper is not None
            and self.rejection_lower > self.rejection_upper
        ):
            raise ValueError("rejection_lower must not exceed rejection_upper.")
        if (
            self.rejection_lower is not None or self.rejection_upper is not None
        ) and not self.use_decoupled_loss:
            raise ValueError("Behavior rejection requires use_decoupled_loss=True.")
        if self.loss_backend == "native" and not self.native_compatible:
            raise ValueError(
                "Native PPO only supports token_mean, token ratios and ordinary "
                "clipping; use custom."
            )

    @property
    def native_compatible(self) -> bool:
        """Return whether PyTRIO's built-in PPO loss can represent this actor."""

        return (
            self.loss_type == "ppo"
            and self.importance_sampling_level == "token"
            and self.loss_aggregation == "token_mean"
            and self.c_clip is None
            and self.rejection_lower is None
            and self.rejection_upper is None
            and not self.use_decoupled_loss
        )


@dataclass(slots=True)
class PPOCriticConfig:
    """Configure the independent PyTRIO critic and its optimizer.

    Parameters:
        enabled (`bool`, default `True`):
            Create and train the critic. Standard PPO enables it; GRPO-family
            reward baselines usually disable it.
        lora_rank (`int` or `None`, default `None`):
            Critic adapter rank. ``None`` reuses the actor rank supplied to the
            trainer.
        optimizer (`OptimizerConfig`, default standard AdamW config):
            Complete independent optimizer settings. When omitted by
            ``PPOConfig``, they are copied from the actor optimizer.
        token_ids (`pair[int]` or `None`, default `None`):
            Positive/negative marker token IDs used for value readout. ``None``
            discovers the single-token encodings of ``"1"`` and ``"0"``.
        value_scale (`float`, default `1.0`):
            Positive multiplier on the marker logprob difference.
        value_bias (`float`, default `0.0`):
            Additive calibration term for every predicted value.
        eps_clip (`float`, default `0.2`):
            Symmetric clipping width for PPO-style value regression.
        n_minibatches (`int`, default `1`):
            Critic optimizer partitions per frozen rollout batch.
        update_epochs (`int`, default `1`):
            Passes over the same frozen return targets.

    The critic is a separate LoRA training client and estimates an unbounded
    scalar from the difference between two marker-token log probabilities.
    """

    enabled: bool = True
    lora_rank: int | None = None
    optimizer: OptimizerConfig | Mapping[str, Any] = field(
        default_factory=OptimizerConfig
    )
    token_ids: tuple[int, int] | list[int] | None = None
    value_scale: float = 1.0
    value_bias: float = 0.0
    eps_clip: float = 0.2
    n_minibatches: int = 1
    update_epochs: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("critic.enabled must be boolean.")
        if self.lora_rank is not None:
            _integer("critic.lora_rank", self.lora_rank, minimum=1)
        self.optimizer = _coerce_optimizer(self.optimizer)
        self.optimizer.__post_init__()
        if self.token_ids is not None and (
            not isinstance(self.token_ids, (tuple, list))
            or len(self.token_ids) != 2
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in self.token_ids
            )
            or self.token_ids[0] == self.token_ids[1]
        ):
            raise ValueError(
                "critic.token_ids must contain two distinct nonnegative token IDs."
            )
        _positive("critic.value_scale", self.value_scale)
        if not math.isfinite(self.value_bias):
            raise ValueError("critic.value_bias must be finite.")
        _nonnegative("critic.eps_clip", self.eps_clip)
        _integer("critic.n_minibatches", self.n_minibatches, minimum=1)
        _integer("critic.update_epochs", self.update_epochs, minimum=1)


@dataclass(init=False)
class PPOConfig(BaseTrainerConfig):
    """Compose a complete PPO-family experiment from functional modules.

    Parameters:
        model (`ModelConfig` or `None`): Trainable actor model and LoRA rank.
        train_dataset (`DatasetConfig` or `None`): Raw prompt training source.
        valid_dataset (`DatasetConfig` or `None`): Optional evaluation source.
        reference (`ReferenceConfig` or `None`): Frozen KL-scoring model. A
            missing reference reuses the actor's unmodified base model.
        training (`TrainingConfig`): Prompt batch size, step limit and seed.
        optimizer (`OptimizerConfig`): AdamW settings for the policy actor.
        checkpoint (`CheckpointConfig`): Periodic actor/critic save cadence.
        evaluation (`EvaluationConfig`): Reward-evaluation schedule. Evaluation
            does not change policy state; it samples the current actor on
            validation prompts and aggregates the configured task reward.
        reporting (`ReportingConfig`): Terminal/SwanLab metric configuration.
        rollout (`PPORolloutConfig`): Prompt rendering and generation requests.
        actor (`PPOActorConfig`): Reward, GAE and clipped policy objective.
        critic (`PPOCriticConfig`): Independent value model and optimizer.

    ``for_algorithm`` applies GRPO-family presets without changing the trainer
    class. ``from_modules`` is the preferred YAML boundary; direct construction
    is useful when a program already owns typed module instances.
    """

    rollout: PPORolloutConfig
    actor: PPOActorConfig
    critic: PPOCriticConfig
    reference: ReferenceConfig | None

    _PPO_ROUTES: ClassVar[dict[str, tuple[str, ...]]] = {
        "num_generations": ("rollout", "num_generations"),
        "max_tokens": ("rollout", "max_tokens"),
        "temperature": ("rollout", "temperature"),
        "top_p": ("rollout", "top_p"),
        "prompt_key": ("rollout", "prompt_key"),
        "chat_template_kwargs": ("rollout", "chat_template_kwargs"),
        **{item.name: ("actor", item.name) for item in fields(PPOActorConfig)},
        "use_critic": ("critic", "enabled"),
        "critic_lora_rank": ("critic", "lora_rank"),
        "critic_learning_rate": ("critic", "optimizer", "learning_rate"),
        "critic_token_ids": ("critic", "token_ids"),
        "critic_value_scale": ("critic", "value_scale"),
        "critic_value_bias": ("critic", "value_bias"),
        "critic_eps_clip": ("critic", "eps_clip"),
        "critic_n_minibatches": ("critic", "n_minibatches"),
        "critic_update_epochs": ("critic", "update_epochs"),
    }

    def __init__(
        self,
        *,
        model=None,
        train_dataset=None,
        valid_dataset=None,
        reference=None,
        training=None,
        optimizer=None,
        checkpoint=None,
        evaluation=None,
        reporting=None,
        rollout=None,
        actor=None,
        critic=None,
        algorithm_plan=None,
        **legacy: Any,
    ) -> None:
        self._init_common(
            training,
            optimizer,
            checkpoint,
            evaluation,
            reporting,
            model=model,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        )
        # Apply shared legacy overrides before constructing the critic. An
        # omitted critic optimizer inherits the final actor optimizer values,
        # including a legacy ``learning_rate=...`` override.
        common_legacy = {
            name: legacy.pop(name)
            for name in tuple(legacy)
            if name in self._COMMON_ROUTES
        }
        self._apply_legacy(common_legacy)
        object.__setattr__(self, "rollout", _coerce_module(rollout, PPORolloutConfig))
        object.__setattr__(self, "actor", _coerce_module(actor, PPOActorConfig))
        object.__setattr__(
            self,
            "reference",
            None
            if reference is None
            else _coerce_module(reference, ReferenceConfig),
        )
        critic_values = dict(critic) if isinstance(critic, Mapping) else critic
        if isinstance(critic_values, dict):
            critic_values.setdefault("optimizer", self.optimizer.optimizer_params())
        elif critic_values is None:
            critic_values = {"optimizer": self.optimizer.optimizer_params()}
        object.__setattr__(
            self, "critic", _coerce_module(critic_values, PPOCriticConfig)
        )
        # Runtime plugins are Python objects and intentionally stay outside
        # YAML serialization; the trainer consumes this optional plan directly.
        self.algorithm_plan = algorithm_plan
        self._apply_legacy(legacy)
        self.__post_init__()

    @classmethod
    def _flat_routes(cls) -> dict[str, tuple[str, ...]]:
        return {**cls._COMMON_ROUTES, **cls._PPO_ROUTES}

    def __post_init__(self) -> None:
        BaseTrainerConfig.__post_init__(self)
        if not isinstance(self.rollout, PPORolloutConfig):
            raise TypeError("rollout must be a PPORolloutConfig.")
        if not isinstance(self.actor, PPOActorConfig):
            raise TypeError("actor must be a PPOActorConfig.")
        if not isinstance(self.critic, PPOCriticConfig):
            raise TypeError("critic must be a PPOCriticConfig.")
        if self.reference is not None and not isinstance(
            self.reference, ReferenceConfig
        ):
            raise TypeError("reference must be a ReferenceConfig or None.")
        self.rollout.__post_init__()
        self.actor.__post_init__()
        self.critic.__post_init__()
        if self.reference is not None:
            self.reference.__post_init__()
        if self.actor.overlong_tokens > self.rollout.max_tokens:
            raise ValueError("overlong_tokens must not exceed rollout.max_tokens.")
        if self.actor.dynamic_sampling and self.rollout.num_generations < 2:
            raise ValueError("Dynamic sampling requires at least two responses per prompt.")
        if self.rollout.temperature != 1.0 or self.rollout.top_p != 1.0:
            raise NotImplementedError(
                "Use temperature=1 and top_p=1: target-only training logprobs "
                "cannot reconstruct temperature/nucleus normalization."
            )

    @property
    def native_compatible(self) -> bool:
        return self.actor.native_compatible

    @classmethod
    def for_algorithm(cls, name: str, **overrides: Any) -> "PPOConfig":
        """Apply a PPO-family preset, then let explicit overrides win."""

        actor_defaults: dict[str, Any] = {"reward_norm": NormConfig()}
        critic_defaults: dict[str, Any] = {"enabled": False}
        if name == "ppo":
            actor_defaults = {"reward_norm": None}
            critic_defaults = {"enabled": True}
        elif name == "grpo":
            pass
        elif name == "drgrpo":
            actor_defaults["reward_norm"] = NormConfig("group", None)
        elif name == "liteppo":
            actor_defaults["reward_norm"] = NormConfig("group", "batch")
        elif name == "rloo":
            actor_defaults["reward_norm"] = NormConfig(
                "group", None, mean_leave1out=True
            )
        elif name == "reinforce":
            actor_defaults["reward_norm"] = None
        elif name == "reinforce_baseline":
            actor_defaults["reward_norm"] = NormConfig("group", None)
        elif name == "gspo":
            actor_defaults["importance_sampling_level"] = "sequence"
        elif name == "sapo":
            actor_defaults["loss_type"] = "sapo"
        elif name == "cispo":
            actor_defaults.update(loss_type="cispo", eps_clip=1.0, eps_clip_higher=4.0)
        elif name == "dapo":
            actor_defaults.update(
                dynamic_sampling=True,
                eps_clip=0.2,
                eps_clip_higher=0.28,
                overlong_tokens=128,
            )
        else:
            raise ValueError(f"Unknown PPO preset: {name!r}.")

        actor = overrides.pop("actor", None)
        if isinstance(actor, Mapping):
            actor = {**actor_defaults, **dict(actor)}
        elif actor is None:
            actor = actor_defaults
        critic = overrides.pop("critic", None)
        if isinstance(critic, Mapping):
            critic = {**critic_defaults, **dict(critic)}
        elif critic is None:
            critic = critic_defaults
        return cls(actor=actor, critic=critic, **overrides)

    @classmethod
    def from_modules(
        cls, settings: Mapping[str, Any], **overrides: Any
    ) -> "PPOConfig":
        """Build a PPO configuration from independent YAML module sections."""

        if not isinstance(settings, Mapping):
            raise TypeError("PPO settings must be a mapping.")
        algorithm = settings.get("algorithm", "grpo")
        if isinstance(algorithm, Mapping):
            algorithm = algorithm.get("name")
        if not isinstance(algorithm, str) or not algorithm.strip():
            raise ValueError("algorithm must be a non-empty string.")
        algorithm = algorithm.strip().lower()

        modules: dict[str, Any] = _common_modules(settings)
        rollout = _module_values(
            PPORolloutConfig,
            settings.get("rollout") or settings.get("gconfig"),
            {
                "group_size": "num_generations",
                "n_samples": "num_generations",
                "max_new_tokens": "max_tokens",
            },
        )
        actor_source = settings.get("actor")
        actor = _module_values(PPOActorConfig, actor_source)
        if isinstance(actor_source, Mapping) and isinstance(
            actor_source.get("optimizer"), Mapping
        ):
            modules["optimizer"].update(
                _module_values(
                    OptimizerConfig,
                    actor_source["optimizer"],
                    {"lr": "learning_rate"},
                )
            )
        critic_source = settings.get("critic")
        critic = _module_values(
            PPOCriticConfig,
            critic_source,
            {
                "learning_rate": "optimizer",
                "lr": "optimizer",
                "critic_learning_rate": "optimizer",
                "critic_lora_rank": "lora_rank",
                "critic_token_ids": "token_ids",
                "critic_value_scale": "value_scale",
                "critic_value_bias": "value_bias",
                "critic_eps_clip": "eps_clip",
                "critic_n_minibatches": "n_minibatches",
                "critic_update_epochs": "update_epochs",
            },
        )
        if isinstance(critic.get("optimizer"), (int, float)):
            critic["optimizer"] = {"learning_rate": critic["optimizer"]}
        if isinstance(critic_source, Mapping) and isinstance(
            critic_source.get("optimizer"), Mapping
        ):
            critic["optimizer"] = _module_values(
                OptimizerConfig,
                critic_source["optimizer"],
                {"lr": "learning_rate"},
            )

        legacy = settings.get(algorithm)
        if isinstance(legacy, Mapping):
            aliases = {
                "group_size": "num_generations",
                "max_new_tokens": "max_tokens",
                "lr": "learning_rate",
            }
            flat = {aliases.get(key, key): value for key, value in legacy.items()}
            overrides = {**flat, **overrides}
        for name in ("reward_norm", "adv_norm"):
            if isinstance(overrides.get(name), Mapping):
                overrides[name] = NormConfig(**dict(overrides[name]))
        modules.update(
            reference=_reference_values(settings.get("reference")),
            rollout=rollout,
            actor=actor,
            critic=critic,
        )
        modules.update(overrides)
        return cls.for_algorithm(algorithm, **modules)


# ---------------------------------------------------------------------------
# DPO modules
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DPODataConfig:
    """Configure preference-pair validation and prompt rendering.

    Parameters:
        max_length (`int` or `None`, default `1024`):
            Maximum prompt-plus-completion token count. DPO rejects overlong
            pairs rather than truncating alternatives asymmetrically; ``None``
            removes the explicit limit.
        prompt_key (`str`, default `"messages"`):
            Dataset field shared by chosen and rejected completions.
        add_eos_token (`bool`, default `True`):
            Append EOS to each completion when absent so preference likelihood
            includes termination behavior.
        chat_template_kwargs (`dict`, default `{}`):
            Model-specific tokenizer template options excluding controls owned
            by preference preparation.
    """

    max_length: int | None = 1024
    prompt_key: str = "messages"
    add_eos_token: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_length is not None:
            _integer("max_length", self.max_length, minimum=2)
        _non_empty_string("prompt_key", self.prompt_key)
        if not isinstance(self.add_eos_token, bool):
            raise ValueError("add_eos_token must be boolean.")
        _chat_template_options(
            self.chat_template_kwargs, {"tokenize", "add_generation_prompt"}
        )


@dataclass(slots=True)
class DPOObjectiveConfig:
    """Configure the pairwise DPO objective.

    Parameters:
        beta (`float`, default `0.1`):
            Positive scale on the policy-versus-reference log-ratio margin.
            Larger values enforce stronger preference separation but can make
            gradients more sensitive to reference-score differences.
    """

    beta: float = 0.1

    def __post_init__(self) -> None:
        _positive("beta", self.beta)


@dataclass(init=False)
class DPOConfig(BaseTrainerConfig):
    """Compose DPO runtime, optimizer, data and objective modules.

    Parameters:
        model (`ModelConfig` or `None`): Trainable policy model and LoRA rank.
        train_dataset (`DatasetConfig` or `None`): Preference training source.
        valid_dataset (`DatasetConfig` or `None`): Optional preference evaluation source.
        reference (`ReferenceConfig` or `None`): Frozen model used to compute
            policy-versus-reference log-ratios. ``None`` reuses the policy base.
        training (`TrainingConfig`): Pair batch size, step limit and seed.
        optimizer (`OptimizerConfig`): AdamW settings for policy updates.
        checkpoint (`CheckpointConfig`): Periodic checkpoint cadence.
        evaluation (`EvaluationConfig`): Preference evaluation schedule.
        reporting (`ReportingConfig`): Metric backend and flush cadence.
        data (`DPODataConfig`): Preference validation and prompt rendering.
        objective (`DPOObjectiveConfig`): Pairwise DPO loss coefficients.
    """

    data: DPODataConfig
    objective: DPOObjectiveConfig
    reference: ReferenceConfig | None

    _DPO_ROUTES: ClassVar[dict[str, tuple[str, ...]]] = {
        **{item.name: ("data", item.name) for item in fields(DPODataConfig)},
        **{item.name: ("objective", item.name) for item in fields(DPOObjectiveConfig)},
    }

    def __init__(
        self, *, model=None, train_dataset=None, valid_dataset=None, reference=None,
        data=None, objective=None, training=None, optimizer=None, checkpoint=None,
        evaluation=None, reporting=None, **legacy
    ):
        self._init_common(
            training, optimizer, checkpoint, evaluation, reporting,
            model=model, train_dataset=train_dataset, valid_dataset=valid_dataset,
            learning_rate=5e-6, project="trio-cookbook-dpo",
        )
        object.__setattr__(
            self,
            "reference",
            None
            if reference is None
            else _coerce_module(reference, ReferenceConfig),
        )
        object.__setattr__(self, "data", _coerce_module(data, DPODataConfig))
        object.__setattr__(self, "objective", _coerce_module(objective, DPOObjectiveConfig))
        self._apply_legacy(legacy)
        self.__post_init__()

    @classmethod
    def _flat_routes(cls):
        return {**cls._COMMON_ROUTES, **cls._DPO_ROUTES}

    def __post_init__(self) -> None:
        BaseTrainerConfig.__post_init__(self)
        if not isinstance(self.data, DPODataConfig):
            raise TypeError("data must be a DPODataConfig.")
        if not isinstance(self.objective, DPOObjectiveConfig):
            raise TypeError("objective must be a DPOObjectiveConfig.")
        if self.reference is not None and not isinstance(
            self.reference, ReferenceConfig
        ):
            raise TypeError("reference must be a ReferenceConfig or None.")
        self.data.__post_init__()
        self.objective.__post_init__()
        if self.reference is not None:
            self.reference.__post_init__()

    @classmethod
    def from_modules(cls, settings: Mapping[str, Any], **overrides: Any):
        """Build DPO modules from YAML sections and legacy ``dpo`` keys."""

        modules = _common_modules(settings)
        legacy = settings.get("dpo")
        if isinstance(legacy, Mapping):
            overrides = {**legacy, **overrides}
        modules["data"] = _module_values(DPODataConfig, settings.get("data"))
        modules["objective"] = _module_values(
            DPOObjectiveConfig, settings.get("objective")
        )
        modules["reference"] = _reference_values(settings.get("reference"))
        modules.update(overrides)
        return cls(**modules)


# ---------------------------------------------------------------------------
# SFT modules
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SFTDataConfig:
    """Configure SFT tokenization, truncation and supervision masks.

    Parameters:
        max_length (`int` or `None`, default `1024`):
            Maximum sequence tokens after right truncation. ``None`` preserves
            the complete sequence.
        dataset_text_field (`str`, default `"text"`):
            Plain-text fallback key when a row has no messages, token IDs, or
            prompt/completion pair.
        completion_only_loss (`bool` or `None`, default `None`):
            Mask prompt tokens in paired examples. ``None`` enables this
            automatically for prompt/completion rows only.
        assistant_only_loss (`bool`, default `False`):
            Train only assistant spans in conversations. Enabling it requires a
            chat template that returns an assistant-token mask.
        add_eos_token (`bool`, default `True`):
            Append EOS to raw non-conversational examples when absent.
        chat_template_kwargs (`dict`, default `{}`):
            Model-specific tokenizer template options excluding return-shape
            and masking arguments owned by SFT preparation.
    """

    max_length: int | None = 1024
    dataset_text_field: str = "text"
    completion_only_loss: bool | None = None
    assistant_only_loss: bool = False
    add_eos_token: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_length is not None:
            _integer("max_length", self.max_length, minimum=2)
        _non_empty_string("dataset_text_field", self.dataset_text_field)
        if self.completion_only_loss is not None and not isinstance(
            self.completion_only_loss, bool
        ):
            raise ValueError("completion_only_loss must be boolean or None.")
        for name in ("assistant_only_loss", "add_eos_token"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        _chat_template_options(
            self.chat_template_kwargs,
            {
                "tokenize",
                "add_generation_prompt",
                "return_dict",
                "return_assistant_tokens_mask",
            },
        )


@dataclass(slots=True)
class SFTExecutionConfig:
    """Bound the number of asynchronous PyTRIO batches in flight.

    Parameters:
        max_in_flight (`int`, default `32`):
            Maximum submitted batches awaiting completion. A larger value can
            improve remote pipeline utilization but retains more batch state in
            local memory.
    """

    max_in_flight: int = 32

    def __post_init__(self) -> None:
        _integer("max_in_flight", self.max_in_flight, minimum=1)


@dataclass(init=False)
class SFTConfig(BaseTrainerConfig):
    """Compose SFT runtime, optimizer, data and async execution modules.

    Parameters:
        model (`ModelConfig` or `None`): Trainable language model and LoRA rank.
        train_dataset (`DatasetConfig` or `None`): Supervised training source.
        valid_dataset (`DatasetConfig` or `None`): Optional loss-evaluation source.
        training (`TrainingConfig`): Sequence batch size, step limit and seed.
        optimizer (`OptimizerConfig`): AdamW settings for language-model updates.
        checkpoint (`CheckpointConfig`): Periodic checkpoint cadence.
        evaluation (`EvaluationConfig`): Token-loss evaluation schedule.
        reporting (`ReportingConfig`): Metric backend and run metadata.
        data (`SFTDataConfig`): Tokenization and supervision-mask policy.
        execution (`SFTExecutionConfig`): Asynchronous request backpressure.
    """

    data: SFTDataConfig
    execution: SFTExecutionConfig

    _SFT_ROUTES: ClassVar[dict[str, tuple[str, ...]]] = {
        **{item.name: ("data", item.name) for item in fields(SFTDataConfig)},
        **{item.name: ("execution", item.name) for item in fields(SFTExecutionConfig)},
    }

    def __init__(
        self, *, model=None, train_dataset=None, valid_dataset=None, data=None,
        execution=None, training=None, optimizer=None, checkpoint=None,
        evaluation=None, reporting=None, **legacy
    ):
        self._init_common(
            training, optimizer, checkpoint, evaluation, reporting,
            model=model, train_dataset=train_dataset, valid_dataset=valid_dataset,
            learning_rate=2e-5, project="trio-cookbook-sft",
        )
        object.__setattr__(self, "data", _coerce_module(data, SFTDataConfig))
        object.__setattr__(self, "execution", _coerce_module(execution, SFTExecutionConfig))
        self._apply_legacy(legacy)
        self.__post_init__()

    @classmethod
    def _flat_routes(cls):
        return {**cls._COMMON_ROUTES, **cls._SFT_ROUTES}

    def __post_init__(self) -> None:
        BaseTrainerConfig.__post_init__(self)
        if not isinstance(self.data, SFTDataConfig):
            raise TypeError("data must be a SFTDataConfig.")
        if not isinstance(self.execution, SFTExecutionConfig):
            raise TypeError("execution must be a SFTExecutionConfig.")
        self.data.__post_init__()
        self.execution.__post_init__()

    @classmethod
    def from_modules(cls, settings: Mapping[str, Any], **overrides: Any):
        """Build SFT modules from YAML sections and legacy ``sft`` keys."""

        modules = _common_modules(settings)
        legacy = settings.get("sft")
        if isinstance(legacy, Mapping):
            overrides = {**legacy, **overrides}
        modules["data"] = _module_values(SFTDataConfig, settings.get("data"))
        modules["execution"] = _module_values(
            SFTExecutionConfig, settings.get("execution")
        )
        modules.update(overrides)
        return cls(**modules)


# ---------------------------------------------------------------------------
# OPD modules
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OPDRolloutConfig:
    """Configure student sampling, sequence limits and prompt rendering.

    Parameters:
        num_generations (`int`, default `1`):
            Student completions sampled for each prompt before teacher scoring.
        max_completion_length (`int`, default `512`):
            Maximum sampled completion tokens.
        max_length (`int` or `None`, default `2048`):
            Total prompt-plus-completion token budget. ``None`` removes the
            explicit total limit while keeping the completion limit.
        prompt_key (`str`, default `"messages"`):
            Dataset field containing the student generation prompt.
        add_generation_prompt (`bool`, default `True`):
            Ask the chat template to append the assistant prefix.
        chat_template_kwargs (`dict`, default `{}`):
            Model-specific prompt-rendering options excluding tokenizer and
            generation-prefix controls owned by OPD.
        stop (`list[str]` or `None`, default `None`):
            Optional nonempty stop strings passed to student sampling. ``None``
            relies on model EOS behavior.
    """

    num_generations: int = 1
    max_completion_length: int = 512
    max_length: int | None = 2048
    prompt_key: str = "messages"
    add_generation_prompt: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    stop: list[str] | None = None

    def __post_init__(self) -> None:
        _integer("num_generations", self.num_generations, minimum=1)
        _integer("max_completion_length", self.max_completion_length, minimum=1)
        if self.max_length is not None:
            _integer("max_length", self.max_length, minimum=2)
        _non_empty_string("prompt_key", self.prompt_key)
        if not isinstance(self.add_generation_prompt, bool):
            raise ValueError("add_generation_prompt must be boolean.")
        _chat_template_options(
            self.chat_template_kwargs, {"tokenize", "add_generation_prompt"}
        )
        if self.stop is not None and (
            not isinstance(self.stop, list)
            or not self.stop
            or any(not isinstance(value, str) or not value for value in self.stop)
        ):
            raise ValueError("stop must be a non-empty list of non-empty strings or None.")


@dataclass(slots=True)
class OPDExecutionConfig:
    """Bound concurrent student sampling and teacher-scoring requests.

    Parameters:
        max_concurrent_sampling (`int`, default `16`):
            Maximum simultaneous student rollout requests.
        max_concurrent_scoring (`int`, default `32`):
            Maximum simultaneous fixed-teacher logprob requests. This may be
            larger because scoring does not autoregressively decode.
    """

    max_concurrent_sampling: int = 16
    max_concurrent_scoring: int = 32

    def __post_init__(self) -> None:
        _integer("max_concurrent_sampling", self.max_concurrent_sampling, minimum=1)
        _integer("max_concurrent_scoring", self.max_concurrent_scoring, minimum=1)


@dataclass(slots=True)
class OPDObjectiveConfig:
    """Configure sampled reverse-KL targets and importance weighting.

    Parameters:
        kl_coef (`float`, default `1.0`):
            Positive multiplier on sampled teacher-minus-student logprob signal.
        discount_factor (`float`, default `0.0`):
            Future-token signal discount. Zero keeps immediate token targets;
            one propagates the full future sum.
        importance_ratio_cap (`float` or `None`, default `5.0`):
            One-sided upper cap on detached current/behavior importance ratios.
            ``None`` restores unbounded importance sampling.
    """

    kl_coef: float = 1.0
    discount_factor: float = 0.0
    importance_ratio_cap: float | None = 5.0

    def __post_init__(self) -> None:
        _positive("kl_coef", self.kl_coef)
        _probability("discount_factor", self.discount_factor, include_zero=True)
        if self.importance_ratio_cap is not None:
            _positive("importance_ratio_cap", self.importance_ratio_cap)


@dataclass(init=False)
class OPDConfig(BaseTrainerConfig):
    """Compose OPD runtime, rollout, concurrency and objective modules.

    Parameters:
        model (`ModelConfig` or `None`): Trainable student model and LoRA rank.
        train_dataset (`DatasetConfig` or `None`): Prompt training source.
        valid_dataset (`DatasetConfig` or `None`): Optional distillation evaluation source.
        teacher (`TeacherConfig` or `None`): Fixed single-teacher identity and
            optional checkpoint. Multi-teacher callers continue to inject specs.
        training (`TrainingConfig`): Prompt batch size, step limit and seed.
        optimizer (`OptimizerConfig`): Student AdamW settings.
        checkpoint (`CheckpointConfig`): Periodic student checkpoint cadence.
        evaluation (`EvaluationConfig`): Distillation evaluation schedule.
        reporting (`ReportingConfig`): Metric backend and aggregation cadence.
        rollout (`OPDRolloutConfig`): Student generation and token budgets.
        execution (`OPDExecutionConfig`): Sampling/scoring concurrency limits.
        objective (`OPDObjectiveConfig`): Reverse-KL and importance weighting.

    Generation distribution parameters are intentionally fixed by the OPD
    trainer so sampled behavior log-probabilities remain aligned with its
    surrogate objective.
    """

    rollout: OPDRolloutConfig
    execution: OPDExecutionConfig
    objective: OPDObjectiveConfig
    teacher: TeacherConfig | None

    _OPD_ROUTES: ClassVar[dict[str, tuple[str, ...]]] = {
        **{item.name: ("rollout", item.name) for item in fields(OPDRolloutConfig)},
        **{item.name: ("execution", item.name) for item in fields(OPDExecutionConfig)},
        **{item.name: ("objective", item.name) for item in fields(OPDObjectiveConfig)},
    }

    def __init__(
        self, *, model=None, train_dataset=None, valid_dataset=None, teacher=None,
        rollout=None, execution=None, objective=None, training=None, optimizer=None,
        checkpoint=None, evaluation=None, reporting=None, **legacy
    ):
        self._init_common(
            training, optimizer, checkpoint, evaluation, reporting,
            model=model, train_dataset=train_dataset, valid_dataset=valid_dataset,
            batch_size=4, project="trio-cookbook-opd",
        )
        object.__setattr__(
            self,
            "teacher",
            None if teacher is None else _coerce_module(teacher, TeacherConfig),
        )
        object.__setattr__(self, "rollout", _coerce_module(rollout, OPDRolloutConfig))
        object.__setattr__(self, "execution", _coerce_module(execution, OPDExecutionConfig))
        object.__setattr__(self, "objective", _coerce_module(objective, OPDObjectiveConfig))
        self._apply_legacy(legacy)
        self.__post_init__()

    @classmethod
    def _flat_routes(cls):
        return {**cls._COMMON_ROUTES, **cls._OPD_ROUTES}

    def __post_init__(self) -> None:
        BaseTrainerConfig.__post_init__(self)
        if not isinstance(self.rollout, OPDRolloutConfig):
            raise TypeError("rollout must be an OPDRolloutConfig.")
        if not isinstance(self.execution, OPDExecutionConfig):
            raise TypeError("execution must be an OPDExecutionConfig.")
        if not isinstance(self.objective, OPDObjectiveConfig):
            raise TypeError("objective must be an OPDObjectiveConfig.")
        if self.teacher is not None and not isinstance(self.teacher, TeacherConfig):
            raise TypeError("teacher must be a TeacherConfig or None.")
        self.rollout.__post_init__()
        self.execution.__post_init__()
        self.objective.__post_init__()
        if self.teacher is not None:
            self.teacher.__post_init__()

    @classmethod
    def from_modules(cls, settings: Mapping[str, Any], **overrides: Any):
        """Build OPD modules from YAML sections and legacy ``opd`` keys."""

        modules = _common_modules(settings)
        legacy = settings.get("opd")
        if isinstance(legacy, Mapping):
            overrides = {**legacy, **overrides}
        modules["rollout"] = _module_values(OPDRolloutConfig, settings.get("rollout"))
        modules["execution"] = _module_values(
            OPDExecutionConfig, settings.get("execution")
        )
        modules["objective"] = _module_values(
            OPDObjectiveConfig, settings.get("objective")
        )
        modules["teacher"] = settings.get("teacher")
        modules.update(overrides)
        return cls(**modules)


__all__ = [
    "BaseTrainerConfig",
    "CheckpointConfig",
    "DatasetConfig",
    "DPOConfig",
    "DPODataConfig",
    "DPOObjectiveConfig",
    "EvaluationConfig",
    "ModelConfig",
    "NormConfig",
    "OPDConfig",
    "OPDExecutionConfig",
    "OPDObjectiveConfig",
    "OPDRolloutConfig",
    "OptimizerConfig",
    "PPOActorConfig",
    "PPOConfig",
    "PPOCriticConfig",
    "PPORolloutConfig",
    "ReportingConfig",
    "ReferenceConfig",
    "SFTConfig",
    "SFTDataConfig",
    "SFTExecutionConfig",
    "TeacherConfig",
    "TrainingConfig",
]
