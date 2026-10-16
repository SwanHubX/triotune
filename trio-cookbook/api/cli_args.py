"""Public configuration and command-line contracts for PyTRIO trainers.

This module is the trainer's contract layer: common run controls live on
``BaseTrainerConfig`` while algorithm subclasses add only their own behavior.
Configuration fields can be supplied directly from recipe/TOML mappings.
Evaluation uses flat eval_* fields instead of the former nested evaluation
object: every_steps becomes eval_steps, before_train/after_train become
eval_before_train/eval_after_train, and batch_size/max_batches become
eval_batch_size/eval_max_batches. Trainers receive eval_dataset separately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .loss_api import POLICY_LOSS_FNS, LossSpec


@dataclass(slots=True)
class BaseTrainerConfig:
    """Common optimizer, lifecycle, and reporting options for every trainer.

    Parameters:
        batch_size (`int`, defaults to `1`):
            Number of dataset examples consumed by one optimizer update. The
            meaning of one example is algorithm-specific: prompt for GRPO/OPD,
            preference pair for DPO, and supervised sequence for SFT.
        max_steps (`int` or `None`, defaults to `None`):
            Maximum optimizer updates. ``None`` consumes the configured dataset
            once, so this limits updates rather than remote requests.
        learning_rate (`float`, defaults to `1e-5`):
            Step size passed to PyTRIO's remote Adam optimizer. Algorithm
            subclasses override the default when their objective needs it.
        beta1 (`float`, defaults to `0.9`):
            Exponential decay for Adam's first-moment estimate.
        beta2 (`float`, defaults to `0.95`):
            Exponential decay for Adam's second-moment estimate.
        eps (`float`, defaults to `1e-8`):
            Positive denominator stabilizer used by Adam.
        weight_decay (`float`, defaults to `0.0`):
            Decoupled AdamW weight decay. Zero disables parameter shrinkage.
        grad_clip_norm (`float`, defaults to `0.0`):
            Global gradient-norm ceiling applied remotely. Zero disables clipping.
        seed (`int` or `None`, defaults to `None`):
            Seed forwarded to PyTRIO clients. ``None`` leaves seed selection to
            the service; algorithms may derive per-request seeds from it.
        save_every (`int`, defaults to `0`):
            Optimizer-step interval for periodic checkpoints. Zero disables
            periodic saves; trainers may still save a final checkpoint.
        report_to (`"swanlab"` or `None`, defaults to `None`):
            Optional metrics backend. ``None`` keeps only terminal progress.
        project (`str`, defaults to `"trio-cookbook"`):
            SwanLab project name. Algorithm subclasses provide distinct defaults.
        workspace (`str` or `None`, defaults to `None`):
            SwanLab workspace; ``None`` uses the account default.
        tags (`list[str]` or `None`, defaults to `None`):
            Optional non-empty SwanLab tags attached to the run.
        run_name (`str` or `None`, defaults to `None`):
            Explicit run name. ``None`` lets the trainer derive one from the model.
        log_dir (`str` or `None`, defaults to `None`):
            Local SwanLab output directory; ``None`` uses the SDK default.
        eval_steps (`int`, defaults to `0`):
            Evaluate eval_dataset after every N completed optimizer updates.
            Zero disables periodic evaluation. Requires a trainer eval_dataset.
        eval_before_train (`bool`, defaults to `False`):
            Evaluate once before the first training update when data is available.
        eval_after_train (`bool`, defaults to `True`):
            Evaluate at the final step unless already evaluated there. Set False
            to evaluate only at eval_steps boundaries.
        eval_batch_size (`int` or `None`, defaults to `None`):
            Evaluation rows per batch; None reuses training batch_size.
        eval_max_batches (`int` or `None`, defaults to `None`):
            Optional batch limit per evaluation pass; None consumes all eval rows.
    """

    # Training loop: one step consumes one algorithm-specific batch.
    batch_size: int = 1
    max_steps: int | None = None

    # Remote Adam optimizer.
    learning_rate: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0

    # Reproducibility and checkpoint cadence.
    seed: int | None = None
    save_every: int = 0

    # Optional experiment reporting.
    report_to: Literal["swanlab"] | None = None
    project: str = "trio-cookbook"
    workspace: str | None = None
    tags: list[str] | None = None
    run_name: str | None = None
    log_dir: str | None = None

    # The training loop evaluates eval_dataset; each algorithm computes its metrics.
    eval_steps: int = 0
    eval_before_train: bool = False
    eval_after_train: bool = True
    eval_batch_size: int | None = None
    eval_max_batches: int | None = None

    def __post_init__(self) -> None:
        """Validate shared invariants before an algorithm validates its fields.

        These checks are kept at the contract boundary because values commonly
        originate in TOML or CLI input, where Python type annotations are not
        enforced. They prevent invalid optimizer calls, empty logging targets,
        and training loops that cannot make progress.
        """

        self._integer("batch_size", minimum=1)
        if self.max_steps is not None:
            self._integer("max_steps", minimum=1)
        self._integer("save_every", minimum=0)
        for name in ("learning_rate", "eps"):
            self._positive(name)
        for name in ("weight_decay", "grad_clip_norm"):
            self._nonnegative(name)
        for name in ("beta1", "beta2"):
            self._adam_decay(name)
        if self.report_to not in (None, "swanlab"):
            raise ValueError("report_to must be 'swanlab' or None.")
        self._non_empty_string("project")
        if self.workspace is not None and (
            not isinstance(self.workspace, str) or not self.workspace.strip()
        ):
            raise ValueError("workspace must be a non-empty string when provided.")
        if self.tags is not None and (
            not isinstance(self.tags, list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in self.tags)
        ):
            raise ValueError("tags must be a list of non-empty strings when provided.")
        self._integer("eval_steps", minimum=0)
        for name in ("eval_before_train", "eval_after_train"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        for name in ("eval_batch_size", "eval_max_batches"):
            if getattr(self, name) is not None:
                self._integer(name, minimum=1)

    def _integer(self, name: str, *, minimum: int) -> None:
        """Validate an integer field with an inclusive lower bound.

        Args:
            name: Dataclass field name to read from this configuration.
            minimum: Smallest accepted value. Booleans are rejected explicitly
                because Python otherwise treats them as integers.
        """

        value = getattr(self, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    def _positive(self, name: str) -> None:
        """Require a numeric field to be finite and strictly positive.

        Args:
            name: Dataclass field name. This helper is used for optimizer scales,
                temperatures and coefficients where zero would erase learning or
                cause an invalid remote request.
        """

        value = getattr(self, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive.")

    def _adam_decay(self, name: str) -> None:
        """Require a finite Adam decay coefficient in ``[0, 1)``.

        Args:
            name: ``beta1`` or ``beta2``. One is excluded because a decay of
                exactly one would prevent the corresponding moment estimate
                from incorporating new gradients.
        """

        value = getattr(self, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value < 1
        ):
            raise ValueError(f"{name} must be finite and in [0, 1).")

    def _nonnegative(self, name: str) -> None:
        """Require a finite numeric field greater than or equal to zero.

        Args:
            name: Dataclass field name for an optional optimizer effect where
                zero has the explicit meaning "disabled".
        """

        value = getattr(self, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative.")

    def _unit_interval(self, name: str, *, include_zero: bool) -> None:
        """Validate a finite coefficient whose upper bound is one.

        Args:
            name: Dataclass field name.
            include_zero: Whether zero is meaningful. Discount factors allow
                zero; nucleus sampling needs a positive probability mass.
        """

        value = getattr(self, name)
        valid_number = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
        in_range = valid_number and (
            0 <= value <= 1 if include_zero else 0 < value <= 1
        )
        if not in_range:
            interval = "[0, 1]" if include_zero else "(0, 1]"
            raise ValueError(f"{name} must be finite and in {interval}.")

    def _non_empty_string(self, name: str) -> None:
        """Require a string field to contain non-whitespace text.

        Args:
            name: Dataclass field name, typically a dataset column or reporting
                identifier used later as a lookup key.
        """

        value = getattr(self, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")

    def _chat_template_options(self, reserved: set[str]) -> None:
        """Validate extra tokenizer arguments without allowing ownership clashes.

        Args:
            reserved: Keyword names controlled by the trainer. Allowing callers
                to repeat them in ``chat_template_kwargs`` would either raise a
                duplicate-keyword error or silently change token/mask alignment.
        """

        if not isinstance(self.chat_template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a dictionary.")
        conflicts = reserved.intersection(self.chat_template_kwargs)
        if conflicts:
            raise ValueError(
                "chat_template_kwargs must not override "
                + ", ".join(sorted(conflicts))
                + "."
            )

    def optimizer_params(self) -> dict[str, float]:
        """Return the four Adam values consumed by the PyTRIO adapter.

        Returns:
            A plain mapping so the client adapter can construct the SDK-specific
            ``AdamParams`` object without coupling configuration to one SDK release.
        """

        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
        }


@dataclass(slots=True)
class _StepLoggingConfig(BaseTrainerConfig):
    """Add interval-based metric aggregation to the shared trainer contract.

    Parameters:
        logging_step (`int`, defaults to `1`):
            Number of optimizer updates accumulated before emitting averaged
            metrics. The final partial window is still flushed by each trainer.
    """

    logging_step: int = 1

    def __post_init__(self) -> None:
        """Validate common controls and require a non-empty logging interval."""

        BaseTrainerConfig.__post_init__(self)
        self._integer("logging_step", minimum=1)


@dataclass(frozen=True, slots=True)
class NormConfig:
    """Describe the population used to center and scale PPO values.

    Parameters:
        mean_level (``"batch"``, ``"group"`` or ``None``, default
            ``"group"``): Population used to compute the centering mean.
        std_level (``"batch"``, ``"group"`` or ``None``, default
            ``"group"``): Population used to compute the standard deviation.
        mean_leave1out (``bool``, default ``False``): For group means, remove
            the current response before computing its baseline. This is the
            leave-one-out baseline used by RLOO-style presets.
        std_unbiased (``bool``, default ``True``): Use the sample standard
            deviation denominator when at least two valid values are present.
        eps (``float``, default ``1e-5``): Positive stabilizer added after
            standard-deviation scaling.
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
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("Normalization eps must be finite and positive.")


@dataclass(slots=True)
class PPOConfig(BaseTrainerConfig):
    """Configure PPO and its shared GRPO-family policy loop.

    The class is defined beside the other public API contracts so TOML, CLI,
    and Python callers receive one validation path.  Presets only select
    algorithm settings; ``PPOTrainer`` remains the single PyTRIO orchestrator.
    Defaults describe PPO with independent actor and critic clients, while
    ``for_algorithm('grpo')`` disables the critic without changing the loop.

    Parameters:
        use_critic (``bool``, default ``True``): Create and train an independent
            PyTRIO critic. Set to ``False`` for GRPO, RLOO, or REINFORCE-style
            presets that derive advantages without value predictions.
        critic_learning_rate (``float`` or ``None``, default ``None``): Critic
            Adam learning rate. ``None`` reuses ``learning_rate``.
        critic_lora_rank (``int`` or ``None``, default ``None``): Critic LoRA
            rank. ``None`` reuses the trainer's actor ``lora_rank``.
        critic_token_ids (``tuple[int, int]`` or ``None``, default ``None``):
            Positive and negative single-token markers used to read values.
            ``None`` discovers the markers by tokenizing ``"1"`` and ``"0"``.
        critic_value_scale (``float``, default ``1.0``): Scale applied to the
            positive-minus-negative marker logprob difference.
        critic_value_bias (``float``, default ``0.0``): Bias added to each
            marker-derived value prediction.
        critic_eps_clip (``float``, default ``0.2``): Clipping range for the
            critic's value regression objective.
        critic_n_minibatches (``int``, default ``1``): Number of critic
            minibatches per rollout update.
        critic_update_epochs (``int``, default ``1``): Number of passes over the
            frozen critic targets.
        num_generations (``int``, default ``4``): Completions sampled per
            prompt group during online rollout collection.
        max_tokens (``int``, default ``512``): Maximum generated completion
            tokens passed to PyTRIO sampling.
        temperature (``float``, default ``1.0``): Sampling temperature. The
            current target-logprob contract only supports the default value.
        top_p (``float``, default ``1.0``): Nucleus sampling probability. The
            current target-logprob contract only supports the default value.
        prompt_key (``str``, default ``"messages"``): Dataset field containing
            plain text, token IDs, or chat messages.
        chat_template_kwargs (``dict``, default ``{}``): Extra arguments passed
            to prompt rendering; trainer-owned alignment arguments cannot be
            overridden.
        reward_norm (``NormConfig`` or ``None``, default ``None``): Normalizes
            scalar outcome rewards before they are assigned to the final active
            completion token.
        adv_norm (``NormConfig`` or ``None``, default
            ``NormConfig("batch", "batch")``): Normalizes token advantages
            after GAE. ``None`` disables advantage normalization.
        reward_bias (``float``, default ``0.0``): Bias added to every raw
            outcome reward before scaling and clipping.
        reward_scaling (``float``, default ``1.0``): Positive multiplier applied
            after ``reward_bias``.
        reward_clip (``float``, default ``20.0``): Symmetric absolute bound for
            shaped outcome rewards.
        discount (``float``, default ``1.0``): GAE reward discount factor.
        gae_lambda (``float``, default ``1.0``): GAE trace-decay factor.
        kl_ctl (``float``, default ``0.0``): Coefficient for the KL penalty
            between policy and fixed-reference completion logprobs. Zero
            disables reference scoring.
        kl_estimator (``"k1"``, ``"k2"`` or ``"k3"``, default ``"k1"``):
            Estimator used to turn policy/reference logprob differences into a
            token reward.
        mask_no_eos_with_zero (``bool``, default ``False``): Replace outcome
            and process rewards with zero for truncated completions while
            retaining KL rewards and truncated critic bootstrap values.
        eps_clip (``float``, default ``0.2``): Lower and default upper clipping
            width for the PPO policy ratio.
        eps_clip_higher (``float`` or ``None``, default ``None``): Optional
            asymmetric upper clipping width. ``None`` reuses ``eps_clip``.
        c_clip (``float`` or ``None``, default ``None``): Optional dual-clip
            coefficient for negative-advantage policy terms.
        loss_type (``"ppo"``, ``"sapo"`` or ``"cispo"``, default ``"ppo"``):
            Policy objective evaluated by the custom PyTRIO loss callback.
        importance_sampling_level (``"token"`` or ``"sequence"``, default
            ``"token"``): Ratio granularity. ``"sequence"`` enables the GSPO
            geometric sequence ratio.
        loss_aggregation (``"token_mean"``, ``"seq_mean"``,
            ``"prompt_mean"`` or ``"constant"``, default ``"token_mean"``):
            Denominator used to reduce policy loss across tokens, responses, or
            prompt groups.
        loss_aggregation_divisor (``float`` or ``None``, default ``None``):
            Fixed divisor for ``loss_aggregation="constant"``.
        use_decoupled_loss (``bool``, default ``False``): Keep rollout behavior
            logprobs separate from a recomputed proximal policy denominator.
        recompute_logprob (``bool``, default ``False``): Recompute current
            policy logprobs before preparing frozen actor targets.
        rejection_lower (``float`` or ``None``, default ``None``): Optional
            lower bound for behavior/proximal rejection filtering.
        rejection_upper (``float`` or ``None``, default ``None``): Optional
            upper bound for behavior/proximal rejection filtering.
        sapo_tau_pos (``float``, default ``1.0``): SAPO temperature for positive
            advantages.
        sapo_tau_neg (``float``, default ``1.05``): SAPO temperature for
            non-positive advantages.
        ppo_n_minibatches (``int``, default ``1``): Actor minibatches per frozen
            rollout update. Prompt groups remain intact within each minibatch.
        update_epochs (``int``, default ``1``): Number of actor passes over the
            frozen rollout targets.
        loss_backend (``"custom"`` or ``"native"``, default ``"custom"``):
            Select the local PyTRIO custom loss bridge or the SDK's built-in
            token-level PPO objective. Native mode has stricter compatibility
            requirements.
        dynamic_sampling (``bool``, default ``False``): Reject equal-reward
            prompt groups and collect replacement groups until the batch is full.
        max_empty_batches (``int``, default ``32``): Maximum consecutive empty
            dynamic-sampling attempts before failing the rollout update.
        overlong_tokens (``int``, default ``0``): Completion-token budget at
            which the optional overlong penalty begins. Zero disables it.
        overlong_penalty_factor (``float``, default ``1.0``): Maximum penalty
            applied by the overlong completion rule.
        eval_after_train (``bool``, default ``False``): PPO evaluation toggle.
            Evaluation is currently unsupported and must remain ``False``;
            shared evaluation fields are rejected by PPO validation.

        Shared optimizer, checkpoint, reporting, seed, and lifecycle fields are
        documented by :class:`BaseTrainerConfig`.
    """

    use_critic: bool = True
    critic_learning_rate: float | None = None
    critic_lora_rank: int | None = None
    critic_token_ids: tuple[int, int] | None = None
    critic_value_scale: float = 1.0
    critic_value_bias: float = 0.0
    critic_eps_clip: float = 0.2
    critic_n_minibatches: int = 1
    critic_update_epochs: int = 1
    num_generations: int = 4
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    prompt_key: str = "messages"
    chat_template_kwargs: dict = field(default_factory=dict)
    reward_norm: NormConfig | None = None
    adv_norm: NormConfig | None = field(
        default_factory=lambda: NormConfig("batch", "batch")
    )
    reward_bias: float = 0.0
    reward_scaling: float = 1.0
    reward_clip: float = 20.0
    discount: float = 1.0
    gae_lambda: float = 1.0
    kl_ctl: float = 0.0
    kl_estimator: str = "k1"
    mask_no_eos_with_zero: bool = False
    eps_clip: float = 0.2
    eps_clip_higher: float | None = None
    c_clip: float | None = None
    loss_type: str = "ppo"
    importance_sampling_level: str = "token"
    loss_aggregation: str = "token_mean"
    loss_aggregation_divisor: float | None = None
    use_decoupled_loss: bool = False
    recompute_logprob: bool = False
    rejection_lower: float | None = None
    rejection_upper: float | None = None
    sapo_tau_pos: float = 1.0
    sapo_tau_neg: float = 1.05
    ppo_n_minibatches: int = 1
    update_epochs: int = 1
    loss_backend: str = "custom"
    dynamic_sampling: bool = False
    max_empty_batches: int = 32
    overlong_tokens: int = 0
    overlong_penalty_factor: float = 1.0
    eval_after_train: bool = False

    def __post_init__(self) -> None:
        """Validate PPO fields after shared optimizer/lifecycle checks."""

        super(PPOConfig, self).__post_init__()
        for name in (
            "use_critic",
            "use_decoupled_loss",
            "recompute_logprob",
            "dynamic_sampling",
            "mask_no_eos_with_zero",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        for name in ("reward_norm", "adv_norm"):
            if getattr(self, name) is not None and not isinstance(
                getattr(self, name), NormConfig
            ):
                raise ValueError(f"{name} must be a NormConfig or None.")
        self._non_empty_string("prompt_key")
        for name in (
            "num_generations",
            "max_tokens",
            "ppo_n_minibatches",
            "update_epochs",
            "max_empty_batches",
            "critic_n_minibatches",
            "critic_update_epochs",
        ):
            self._integer(name, minimum=1)
        self._integer("overlong_tokens", minimum=0)
        if self.critic_learning_rate is not None:
            self._positive("critic_learning_rate")
        if self.critic_lora_rank is not None:
            self._integer("critic_lora_rank", minimum=1)
        self._positive("critic_value_scale")
        self._nonnegative("critic_eps_clip")
        if not math.isfinite(self.critic_value_bias):
            raise ValueError("critic_value_bias must be finite.")
        if self.critic_token_ids is not None and (
            not isinstance(self.critic_token_ids, (tuple, list))
            or len(self.critic_token_ids) != 2
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in self.critic_token_ids
            )
            or self.critic_token_ids[0] == self.critic_token_ids[1]
        ):
            raise ValueError(
                "critic_token_ids must contain two distinct nonnegative token IDs "
                "(positive, negative)."
            )
        for name in (
            "temperature",
            "reward_scaling",
            "reward_clip",
            "sapo_tau_pos",
            "sapo_tau_neg",
        ):
            self._positive(name)
        for name in ("eps_clip", "kl_ctl", "overlong_penalty_factor"):
            self._nonnegative(name)
        for name in ("top_p", "discount", "gae_lambda"):
            value = getattr(self, name)
            if (
                not math.isfinite(value)
                or not 0 <= value <= 1
                or (name == "top_p" and value == 0)
            ):
                raise ValueError(f"{name} is outside its probability range.")
        if not math.isfinite(self.reward_bias) or self.eps_clip > 1:
            raise ValueError("reward_bias must be finite and eps_clip <= 1.")
        for name in ("eps_clip_higher", "rejection_lower", "rejection_upper"):
            if getattr(self, name) is not None:
                self._positive(name)
        if self.c_clip is not None and (
            not math.isfinite(self.c_clip) or self.c_clip <= 1
        ):
            raise ValueError("c_clip must be finite and > 1.")
        choices = {
            "kl_estimator": ("k1", "k2", "k3"),
            "loss_type": ("ppo", "sapo", "cispo"),
            "importance_sampling_level": ("token", "sequence"),
            "loss_aggregation": ("token_mean", "seq_mean", "prompt_mean", "constant"),
            "loss_backend": ("custom", "native"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}.")
        if self.loss_aggregation == "constant":
            self._positive("loss_aggregation_divisor")
        if self.loss_type == "sapo" and self.use_decoupled_loss:
            raise ValueError("SAPO requires use_decoupled_loss=False.")
        if self.loss_type == "cispo" and (
            self.importance_sampling_level != "token" or self.eps_clip_higher is None
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
        if self.overlong_tokens > self.max_tokens:
            raise ValueError("overlong_tokens must not exceed max_tokens.")
        if self.dynamic_sampling and self.num_generations < 2:
            raise ValueError(
                "Dynamic sampling requires at least two responses per prompt."
            )
        if (
            self.loss_aggregation != "constant"
            and self.loss_aggregation_divisor is not None
        ):
            raise ValueError(
                "loss_aggregation_divisor only applies to constant reduction."
            )
        if self.loss_backend == "native" and not self.native_compatible:
            raise ValueError(
                "Native PPO only supports token_mean, token ratios and ordinary "
                "clipping; use custom."
            )
        if self.eval_steps or self.eval_before_train or self.eval_after_train:
            raise NotImplementedError(
                "This PPO package does not implement scheduled evaluation."
            )
        if self.temperature != 1.0 or self.top_p != 1.0:
            raise NotImplementedError(
                "Use temperature=1 and top_p=1: target-only training logprobs "
                "cannot reconstruct temperature/nucleus normalization."
            )

    @property
    def native_compatible(self) -> bool:
        """Return whether the request matches PyTRIO's built-in PPO contract."""

        return (
            self.loss_type == "ppo"
            and self.importance_sampling_level == "token"
            and self.loss_aggregation == "token_mean"
            and self.c_clip is None
            and self.rejection_lower is None
            and self.rejection_upper is None
            and not self.use_decoupled_loss
        )

    @classmethod
    def for_algorithm(cls, name: str, **overrides) -> "PPOConfig":
        """Return one shared PyTRIO loop configured for a PPO-family preset."""

        options = {"use_critic": False, "reward_norm": NormConfig()}
        if name == "ppo":
            options = {"use_critic": True, "reward_norm": None}
        elif name == "grpo":
            pass
        elif name == "drgrpo":
            options["reward_norm"] = NormConfig("group", None)
        elif name == "liteppo":
            options["reward_norm"] = NormConfig("group", "batch")
        elif name == "rloo":
            options["reward_norm"] = NormConfig("group", None, mean_leave1out=True)
        elif name == "reinforce":
            options["reward_norm"] = None
        elif name == "reinforce_baseline":
            options["reward_norm"] = NormConfig("group", None)
        elif name == "gspo":
            options["importance_sampling_level"] = "sequence"
        elif name == "sapo":
            options["loss_type"] = "sapo"
        elif name == "cispo":
            options.update(loss_type="cispo", eps_clip=1.0, eps_clip_higher=4.0)
        elif name == "dapo":
            options.update(
                dynamic_sampling=True,
                eps_clip=0.2,
                eps_clip_higher=0.28,
                overlong_tokens=128,
            )
        else:
            raise ValueError(f"Unknown PPO-family preset: {name!r}.")
        return cls(**{**options, **overrides})


@dataclass(slots=True)
class GRPOConfig(_StepLoggingConfig):
    """Configure grouped rollout generation and GRPO policy updates.

    Parameters:
        learning_rate (`float`, defaults to `4e-5`):
            GRPO actor learning rate, overriding the conservative shared default.
        project (`str`, defaults to `"trio-cookbook-grpo"`):
            Default SwanLab namespace for GRPO runs.
        output_dir (`bool`, defaults to `False`):
            Log the final checkpoint URI as an output artifact when reporting.
        num_generations (`int`, defaults to `8`):
            Completions sampled per prompt. At least two are required because
            GRPO centers rewards within each prompt group.
        max_completion_length (`int`, defaults to `512`):
            Maximum generated tokens for each completion.
        temperature (`float`, defaults to `1.0`):
            Sampling temperature. It must stay positive so rollout logprobs
            describe a valid stochastic behavior policy.
        top_p (`float`, defaults to `1.0`):
            Nucleus probability threshold in ``(0, 1]``.
        top_k (`int`, defaults to `-1`):
            Top-k cutoff; ``-1`` disables top-k filtering in PyTRIO.
        loss_fn (`str`, defaults to `"importance_sampling"`):
            PyTRIO policy objective: ``importance_sampling``, ``ppo``, ``cispo``
            or ``dro``. Rollout and group-centered advantage semantics remain GRPO.
        loss_fn_config (`dict[str, Any]`, defaults to `{}`):
            Additional options forwarded to the remote loss implementation.
        reward_weights (`list[float]` or `None`, defaults to `None`):
            Per-reward-function coefficients. ``None`` assigns weight one.
        scale_rewards (`"none"`, `"group"`, or `"batch"`, defaults to `"none"`):
            Standard-deviation normalization applied after group centering.
        reward_epsilon (`float`, defaults to `1e-4`):
            Positive stabilizer added to the reward standard deviation.
        skip_zero_advantage (`bool`, defaults to `True`):
            Omit groups whose centered advantages are all zero.
        prompt_key (`str`, defaults to `"messages"`):
            Dataset column containing text, token IDs or chat messages.
        add_generation_prompt (`bool`, defaults to `True`):
            Ask the chat template to append the assistant-generation prefix.
        stop (`bool`, defaults to `True`):
            Stop at tokenizer EOS/chat-end markers. ``False`` relies only on the
            completion token budget.
        chat_template_kwargs (`dict[str, Any]`, defaults to `{}`):
            Extra chat-template options. Tokenization and generation-prefix
            control remain owned by the trainer to preserve alignment.

        Shared parameters are documented by :class:`BaseTrainerConfig` and
        ``logging_step`` by :class:`_StepLoggingConfig`.
    """

    # Algorithm-specific overrides of shared defaults.
    learning_rate: float = 4e-5
    project: str = "trio-cookbook-grpo"
    output_dir: bool = False

    # Rollout distribution and completion budget.
    num_generations: int = 8
    max_completion_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    # Remote objective and reward-to-advantage transformation.
    loss_fn: str = "importance_sampling"
    loss_fn_config: dict[str, Any] = field(default_factory=dict)
    reward_weights: list[float] | None = None
    scale_rewards: str = "none"
    reward_epsilon: float = 1e-4
    skip_zero_advantage: bool = True
    # Dataset column and tokenizer rendering controls.
    prompt_key: str = "messages"
    add_generation_prompt: bool = True
    stop: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate sampling, reward scaling and remote-loss compatibility."""

        _StepLoggingConfig.__post_init__(self)
        self._integer("num_generations", minimum=2)
        self._integer("max_completion_length", minimum=1)
        if self.top_k != -1:
            self._integer("top_k", minimum=1)
        self._unit_interval("top_p", include_zero=False)
        self._positive("temperature")
        if self.scale_rewards not in {"none", "group", "batch"}:
            raise ValueError("scale_rewards must be 'none', 'group', or 'batch'.")
        self._positive("reward_epsilon")
        self._non_empty_string("prompt_key")
        if not isinstance(self.stop, bool):
            raise TypeError("stop must be a boolean.")
        if not isinstance(self.output_dir, bool):
            raise TypeError("output_dir must be a boolean.")
        self._chat_template_options({"tokenize", "add_generation_prompt"})
        loss = LossSpec(self.loss_fn, self.loss_fn_config)
        if loss.name not in POLICY_LOSS_FNS:
            raise ValueError("GRPO requires a PyTRIO policy loss.")
        self.loss_fn = loss.name
        self.loss_fn_config = loss.loss_fn_config

    @property
    def sampling_params(self) -> dict[str, Any]:
        """Translate generation fields to the PyTRIO sampling vocabulary.

        Returns:
            SDK-neutral keyword arguments. Stop sequences are model-dependent
            and are added later by ``GRPOTrainer``.
        """

        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_completion_length,
            "seed": self.seed,
        }


@dataclass(slots=True)
class DPOConfig(_StepLoggingConfig):
    """Configure pairwise preference optimization with a fixed reference model.

    Parameters:
        learning_rate (`float`, defaults to `5e-6`):
            Conservative actor learning rate for the pairwise DPO objective.
        project (`str`, defaults to `"trio-cookbook-dpo"`):
            Default SwanLab namespace for DPO runs.
        beta (`float`, defaults to `0.1`):
            Positive scale applied to the policy-versus-reference log-ratio
            margin before the logistic DPO loss.
        max_length (`int` or `None`, defaults to `1024`):
            Maximum prompt-plus-completion length. DPO rejects overlong pairs
            rather than truncating one preference differently from the other.
        prompt_key (`str`, defaults to `"messages"`):
            Dataset column shared by chosen and rejected completions.
        add_eos_token (`bool`, defaults to `True`):
            Append tokenizer EOS to each completion when absent so preference
            likelihood includes response termination.
        chat_template_kwargs (`dict[str, Any]`, defaults to `{}`):
            Extra chat-template options. Tokenization and generation-prefix
            arguments remain controlled by dataset preparation.

        Shared parameters are documented by :class:`BaseTrainerConfig` and
        ``logging_step`` by :class:`_StepLoggingConfig`.
    """

    # Algorithm-specific overrides of shared defaults.
    learning_rate: float = 5e-6
    project: str = "trio-cookbook-dpo"

    # Preference objective and pair length contract.
    beta: float = 0.1
    max_length: int | None = 1024
    # Dataset column and completion rendering controls.
    prompt_key: str = "messages"
    add_eos_token: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate preference length, KL scale and prompt rendering options."""

        _StepLoggingConfig.__post_init__(self)
        self._positive("beta")
        if self.max_length is not None:
            self._integer("max_length", minimum=2)
        self._non_empty_string("prompt_key")
        self._chat_template_options({"tokenize", "add_generation_prompt"})


@dataclass(slots=True)
class SFTConfig(BaseTrainerConfig):
    """Configure text-only causal-language-model supervised fine-tuning.

    Parameters:
        learning_rate (`float`, defaults to `2e-5`):
            SFT actor learning rate, between the conservative preference and
            higher-variance policy-gradient defaults.
        project (`str`, defaults to `"trio-cookbook-sft"`):
            Default SwanLab namespace for SFT runs.
        max_in_flight (`int`, defaults to `32`):
            Maximum submitted batches awaiting completion. This bounds memory
            while allowing PyTRIO forward/backward and optimizer requests to
            remain pipelined.
        max_length (`int` or `None`, defaults to `1024`):
            Maximum tokens per sequence. SFT truncates from the right and then
            verifies that at least one next-token label remains.
        dataset_text_field (`str`, defaults to `"text"`):
            Plain-text fallback column when a row has no messages, pair or
            pretokenized input.
        completion_only_loss (`bool` or `None`, defaults to `None`):
            ``True`` masks prompt tokens in prompt/completion rows. ``None``
            enables that behavior automatically only for paired rows.
        assistant_only_loss (`bool`, defaults to `False`):
            Train only assistant spans in conversational rows. The tokenizer
            chat template must provide an assistant-token mask.
        add_eos_token (`bool`, defaults to `True`):
            Append tokenizer EOS to un-tokenized, non-conversational sequences.
        chat_template_kwargs (`dict[str, Any]`, defaults to `{}`):
            Extra template options. Return shape and assistant-mask arguments
            remain controlled by dataset preparation.

        Shared parameters are documented by :class:`BaseTrainerConfig`.
    """

    # Algorithm-specific overrides of shared defaults.
    learning_rate: float = 2e-5
    project: str = "trio-cookbook-sft"

    # Async pipeline capacity and sequence preprocessing.
    max_in_flight: int = 32
    max_length: int | None = 1024
    dataset_text_field: str = "text"
    completion_only_loss: bool | None = None
    assistant_only_loss: bool = False
    add_eos_token: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate pipeline capacity, sequence budget and tokenizer ownership."""

        BaseTrainerConfig.__post_init__(self)
        self._integer("max_in_flight", minimum=1)
        if self.max_length is not None:
            self._integer("max_length", minimum=2)
        self._non_empty_string("dataset_text_field")
        self._chat_template_options(
            {
                "tokenize",
                "add_generation_prompt",
                "return_dict",
                "return_assistant_tokens_mask",
            }
        )


@dataclass(slots=True)
class OPDConfig(_StepLoggingConfig):
    """Configure sampled, token-level reverse-KL distillation.

    Parameters:
        batch_size (`int`, defaults to `4`):
            Prompts per optimizer update.
        max_steps (`int` or `None`, defaults to `None`):
            Optimizer step limit.
        learning_rate (`float`, defaults to `1e-5`):
            Remote Adam learning rate.
        beta1 (`float`, defaults to `0.9`):
            Adam first-moment coefficient.
        beta2 (`float`, defaults to `0.95`):
            Adam second-moment coefficient.
        eps (`float`, defaults to `1e-8`):
            Adam denominator smoothing value.
        weight_decay (`float`, defaults to `0.0`):
            AdamW decay coefficient.
        grad_clip_norm (`float`, defaults to `0.0`):
            Gradient clipping ceiling.
        seed (`int` or `None`, defaults to `None`):
            Sampling seed base.
        save_every (`int`, defaults to `0`):
            Periodic checkpoint interval.
        report_to (`"swanlab"` or `None`, defaults to `None`):
            Reporting backend.
        project (`str`, defaults to `"trio-cookbook-opd"`):
            SwanLab project.
        workspace (`str` or `None`, defaults to `None`):
            SwanLab workspace.
        tags (`list[str]` or `None`, defaults to `None`):
            SwanLab tags.
        run_name (`str` or `None`, defaults to `None`):
            SwanLab run name.
        log_dir (`str` or `None`, defaults to `None`):
            SwanLab log directory.
        eval_steps (`int`, defaults to `0`):
            Evaluation interval in optimizer steps.
        eval_before_train (`bool`, defaults to `False`):
            Evaluate before training.
        eval_after_train (`bool`, defaults to `True`):
            Evaluate the final step once.
        eval_batch_size (`int` or `None`, defaults to `None`):
            Evaluation batch size.
        eval_max_batches (`int` or `None`, defaults to `None`):
            Evaluation batch limit.
        logging_step (`int`, defaults to `1`):
            Metric aggregation interval.
        num_generations (`int`, defaults to `1`):
            Completions per prompt.
        max_completion_length (`int`, defaults to `512`):
            Completion token limit.
        max_length (`int` or `None`, defaults to `2048`):
            Total token budget.
        max_concurrent_sampling (`int`, defaults to `16`):
            Sampling concurrency.
        max_concurrent_scoring (`int`, defaults to `32`):
            Teacher concurrency.
        kl_coef (`float`, defaults to `1.0`):
            Reverse-KL signal coefficient.
        discount_factor (`float`, defaults to `0.0`):
            Future-signal discount.
        importance_ratio_cap (`float` or `None`, defaults to `5.0`):
            One-sided upper bound for the detached current/behavior importance
            ratio. ``None`` restores unbounded importance sampling.
        prompt_key (`str`, defaults to `"messages"`):
            Dataset prompt column.
        add_generation_prompt (`bool`, defaults to `True`):
            Add assistant prefix.
        chat_template_kwargs (`dict[str, Any]`, defaults to `{}`):
            Template options.
        stop (`list[str]` or `None`, defaults to `None`):
            Sampling stop strings.
    """

    # Algorithm-specific overrides of shared defaults.
    batch_size: int = 4
    project: str = "trio-cookbook-opd"

    # Student rollout and teacher-scoring concurrency.
    num_generations: int = 1
    max_completion_length: int = 512
    max_length: int | None = 2048
    max_concurrent_sampling: int = 16
    max_concurrent_scoring: int = 32
    # Sampled reverse-KL signal and temporal credit assignment.
    kl_coef: float = 1.0
    discount_factor: float = 0.0
    importance_ratio_cap: float | None = 5.0
    # Prompt rendering and generation termination.
    prompt_key: str = "messages"
    add_generation_prompt: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    stop: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate rollout concurrency and sampled reverse-KL semantics.

        OPD deliberately fixes temperature/top-p/top-k inside the trainer so
        sampled logprobs match the behavior distribution used by its surrogate;
        only stop strings and token budgets are configurable here.
        """

        _StepLoggingConfig.__post_init__(self)
        for name in (
            "num_generations",
            "max_completion_length",
            "max_concurrent_sampling",
            "max_concurrent_scoring",
        ):
            self._integer(name, minimum=1)
        if self.max_length is not None:
            self._integer("max_length", minimum=2)
        if self.seed is not None:
            self._integer("seed", minimum=0)
        self._positive("kl_coef")
        self._unit_interval("discount_factor", include_zero=True)
        if self.importance_ratio_cap is not None:
            self._positive("importance_ratio_cap")
        self._non_empty_string("prompt_key")
        if not isinstance(self.add_generation_prompt, bool):
            raise ValueError("add_generation_prompt must be boolean.")
        self._chat_template_options({"tokenize", "add_generation_prompt"})
        if self.stop is not None and (
            not isinstance(self.stop, list)
            or not self.stop
            or any(not isinstance(value, str) or not value for value in self.stop)
        ):
            raise ValueError("stop must be a non-empty list of non-empty strings or None.")


__all__ = [
    "BaseTrainerConfig",
    "DPOConfig",
    "GRPOConfig",
    "NormConfig",
    "OPDConfig",
    "PPOConfig",
    "SFTConfig",
]
