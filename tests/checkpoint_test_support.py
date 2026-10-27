"""Stateful CPU model/Adam service used to audit real trainer continuation.

The transport is in memory; losses and Adam moments are actual Torch tensors.
Saving copies weights and optimizer state, and loading creates an independent
client. This detects a weights-only restore even when update counters agree.
"""

import copy
import random
from types import SimpleNamespace

import numpy as np
import torch
from agent_test_support import AgentTokenizer, CalculatorSampler
from test_sft_trainer import _Tokenizer


class Tokenizer(_Tokenizer):
    def get_vocab(self):
        return {chr(i): i for i in range(256)}

    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens if t != 255)


def token_logprobs(weight, targets):
    coefficients = torch.tensor(
        [(int(t) % 7 + 1) / 7 for t in targets], dtype=torch.float64
    )
    return torch.nn.functional.logsigmoid(weight * coefficients)


class Sampler:
    def __init__(self, service, weight, version):
        self.service, self.weight, self.version = (
            service,
            weight.detach().clone(),
            version,
        )

    def get_tokenizer(self):
        return Tokenizer()

    def compute_logprobs(self, prompt):
        ids = prompt.to_ints()
        return [None] + token_logprobs(self.weight, ids[1:]).tolist()

    async def compute_logprobs_async(self, prompt):
        return self.compute_logprobs(prompt)

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        ids = prompt.to_ints()
        self.service.samples.append((ids, sampling_params.seed, self.version))
        if self.service.agent:
            raw = await CalculatorSampler(
                AgentTokenizer(), [], self.version
            ).sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=sampling_params,
            )
            sequence = raw.sequences[0]
            sequence.logprobs = token_logprobs(self.weight, sequence.tokens).tolist()
            return raw
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=[65 + index, 1],
                    logprobs=token_logprobs(self.weight, [65 + index, 1]).tolist(),
                    text=chr(65 + index),
                    stop_reason="stop",
                )
                for index in range(num_samples)
            ]
        )


class Client:
    def __init__(self, service, saved=None):
        self.service = service
        self.weight = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        self.optimizer = torch.optim.AdamW([self.weight], lr=0.001)
        self.steps = 0
        self.inputs, self.lrs = [], []
        self.random_draws = []
        if saved is not None:
            self.weight.data.copy_(saved["weight"])
            self.optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
            self.steps = saved["steps"]

    def get_tokenizer(self):
        return AgentTokenizer() if self.service.agent else Tokenizer()

    def _forward(
        self,
        data,
        *,
        backward=False,
        loss_fn="cross_entropy",
        loss_fn_config=None,
        **kwargs,
    ):
        targets = [list(d.loss_fn_inputs["target_tokens"].data) for d in data]
        rows = [token_logprobs(self.weight, values) for values in targets]
        masks = [torch.tensor([t != -100 for t in values]) for values in targets]
        count = sum(int(mask.sum()) for mask in masks)
        if callable(loss_fn):
            loss, metrics = loss_fn(data, rows)
        elif loss_fn == "cross_entropy":
            loss = -torch.cat([row[mask] for row, mask in zip(rows, masks)]).mean()
            metrics = {"loss_mean": loss.item(), "loss_sum": loss.item() * count}
        else:
            terms = []
            for datum, row, mask in zip(data, rows, masks):
                fields = datum.loss_fn_inputs
                old = torch.tensor(fields["logprobs"].data)
                advantages = torch.tensor(fields["advantages"].data)
                terms.append(-(torch.exp(row - old) * advantages)[mask])
            loss = torch.cat(terms).mean()
            metrics = {"loss_mean": loss.item()}
        if backward:
            self.inputs.append([d.model_input.to_ints() for d in data])
            self.optimizer.zero_grad()
            loss.backward()
        return SimpleNamespace(
            metrics={**metrics, "token_count": count},
            loss_fn_outputs=[{"logprobs": row.detach()} for row in rows],
        )

    def forward(self, data, **kwargs):
        return self._forward(data, **kwargs)

    async def forward_async(self, data, **kwargs):
        return self.forward(data, **kwargs)

    async def forward_backward_async(self, data, **kwargs):
        return self._forward(data, backward=True, **kwargs)

    async def forward_backward_custom_async(self, data, loss_fn, **kwargs):
        return self._forward(data, backward=True, loss_fn=loss_fn, **kwargs)

    def forward_backward_custom(self, data, loss_fn, **kwargs):
        return self._forward(data, backward=True, loss_fn=loss_fn, **kwargs)

    def optim_step(self, params):
        self.random_draws.append(
            (random.random(), float(np.random.random()), float(torch.rand(())))
        )
        self.lrs.append(params.learning_rate)
        for group in self.optimizer.param_groups:
            group.update(
                lr=params.learning_rate,
                betas=(params.beta1, params.beta2),
                eps=params.eps,
                weight_decay=params.weight_decay,
            )
        self.optimizer.step()
        self.steps += 1
        return SimpleNamespace(metrics={})

    async def optim_step_async(self, params):
        return self.optim_step(params)

    def save_state(self, *, name):
        if self.service.fail_save:
            raise RuntimeError("test save failed")
        path = f"trio://{len(self.service.saved)}/{name}"
        self.service.saved[path] = {
            "weight": self.weight.detach().clone(),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "steps": self.steps,
        }
        return SimpleNamespace(path=path)

    async def save_state_async(self, **kwargs):
        return self.save_state(**kwargs)

    async def save_weights_and_get_sampling_client_async(self):
        return Sampler(self.service, self.weight, self.steps)


class Service:
    def __init__(self, *, agent=False):
        self.agent = agent
        self.saved, self.clients, self.loaded, self.samples = {}, [], [], []
        self.fail_save = False

    def create_lora_training_client(self, **kwargs):
        client = Client(self)
        self.clients.append(client)
        return client

    async def create_lora_training_client_async(self, **kwargs):
        return self.create_lora_training_client(**kwargs)

    def create_training_client_from_state_with_optimizer(self, *, path):
        self.loaded.append(path)
        client = Client(self, self.saved[path])
        self.clients.append(client)
        return client

    async def create_training_client_from_state_with_optimizer_async(self, **kwargs):
        return self.create_training_client_from_state_with_optimizer(**kwargs)

    def create_sampling_client(self, **kwargs):
        return Sampler(self, torch.tensor(0.5, dtype=torch.float64), -1)
