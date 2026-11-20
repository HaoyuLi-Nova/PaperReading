# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Adapted for the World2Motion inference release.
from __future__ import annotations

from typing import Callable, Optional
import attrs
import torch
from scripts.utils.config import make_freezable
from scripts.utils.progress_bar import progress_bar
from scripts.sampling.fm_solvers_unipc import FlowUniPCMultistepScheduler
from scripts.sampling.utils import run_multiseed


@make_freezable
@attrs.define(slots=False)
class UniPCSamplerConfig:
    num_train_timesteps: int = 1000
    shift: float = 1.0
    use_dynamic_shifting: bool = False


class UniPCSampler(torch.nn.Module):
    def __init__(self, cfg: Optional[UniPCSamplerConfig] = None, tensor_kwargs: Optional[dict] = None):
        super().__init__()
        if cfg is None:
            cfg = UniPCSamplerConfig()
        self.cfg = cfg
        self.tensor_kwargs = tensor_kwargs

    @torch.no_grad()
    def forward(
        self,
        velocity_fn: Callable,
        noise: torch.Tensor | list[torch.Tensor],
        num_steps: int = 35,
        shift: float | None = None,
        seed: int | list[int] | None = None,
        shift_action: float | None = None,
        action_spans: list[tuple[int, int] | None] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run the UniPC multi-step sampling loop.

        ``noise`` and ``seed`` must both be single values or both be lists
        (of the same length).  When lists are provided, each element
        corresponds to one independent sample with its own RNG generator
        and scheduler; the return value is then a list of denoised tensors.
        When single values are provided, a single tensor is returned.

        Args:
            velocity_fn: ``velocity_fn(noise=..., timestep=...) -> velocity``.
                Under the dual-schedule mode (``shift_action`` set) it is called as
                ``velocity_fn(noise, timestep, timestep_action=...)`` so the model can
                embed the action segment's own timestep.
            noise: Initial noise.  Either a single ``torch.Tensor`` of shape
                ``(C, T, H, W)`` or a ``list[torch.Tensor]`` where each
                element has shape ``(C, T, H, W)``.
            seed: RNG seed.  Either a single ``int`` or a ``list[int]`` with
                the same length as ``noise``.
            num_steps: Number of denoising steps.
            shift: Flow-matching shift factor.  Defaults to ``self.cfg.shift``.
            shift_action: Optional flow-matching shift for the ACTION segment of each
                flat latent. When set (dual-schedule / modality-forcing mode), the
                action segment follows its own UniPC trajectory built with this shift,
                while the rest of the latent keeps ``shift``. Requires ``action_spans``
                and list-form ``noise``/``seed``.
            action_spans: Per-sample ``(offset, length)`` of the action segment inside
                the flat latent (``None`` for samples without action). The UniPC update
                is elementwise, so stepping the two segments with separate schedulers
                is exact.

        Returns:
            Denoised sample(s).  A single ``torch.Tensor`` when ``noise`` is a
            tensor, or a ``list[torch.Tensor]`` when ``noise`` is a list.
        """
        if shift is None:
            shift = self.cfg.shift
        assert isinstance(shift, float), "Shift must be a float"
        if shift_action is not None:
            assert action_spans is not None, "shift_action requires action_spans"
            assert isinstance(noise, list) and isinstance(seed, list), (
                "dual-schedule mode requires list-form noise/seed"
            )
            return self._forward_dual_schedule(
                velocity_fn=velocity_fn,
                noise=noise,
                num_steps=num_steps,
                shift=float(shift),
                shift_action=float(shift_action),
                action_spans=action_spans,
                seed=seed,
            )

        def _init_sample_scheduler(seed: int | None) -> tuple[torch.Generator, FlowUniPCMultistepScheduler]:
            seed_g = torch.Generator(device=self.tensor_kwargs["device"])
            if seed is not None:
                seed_g.manual_seed(seed)
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.cfg.num_train_timesteps,
                shift=self.cfg.shift,
                use_dynamic_shifting=self.cfg.use_dynamic_shifting,
            )
            sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift)
            return (seed_g, sample_scheduler)

        seed_g, sample_scheduler = run_multiseed(_init_sample_scheduler, seed=seed)
        timesteps = sample_scheduler[0].timesteps if isinstance(sample_scheduler, list) else sample_scheduler.timesteps
        latent = noise
        for timestep in progress_bar(timesteps, desc="Sampling", total=len(timesteps)):
            velocity_pred = velocity_fn(latent, timestep.reshape(1, 1))

            def _scheduler_step(
                seed_g: torch.Generator,
                sample_scheduler: FlowUniPCMultistepScheduler,
                velocity_pred: torch.Tensor,
                latent: torch.Tensor,
            ) -> torch.Tensor:
                return sample_scheduler.step(
                    model_output=velocity_pred,
                    timestep=timestep,
                    sample=latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0].squeeze(0)

            latent = run_multiseed(
                _scheduler_step,
                seed_g=seed_g,
                sample_scheduler=sample_scheduler,
                velocity_pred=velocity_pred,
                latent=latent,
            )
        return latent

    def _forward_dual_schedule(
        self,
        velocity_fn: Callable,
        noise: list[torch.Tensor],
        num_steps: int,
        shift: float,
        shift_action: float,
        action_spans: list[tuple[int, int] | None],
        seed: list[int | None],
    ) -> list[torch.Tensor]:
        """Dual-schedule UniPC loop: the action segment of each flat latent follows its
        own scheduler (``shift_action``) while the remainder keeps ``shift``.

        Both schedulers are stepped once per iteration over the same ``num_steps`` grid,
        so they reach sigma=0 together; only the sigma trajectory in between differs.
        ``FlowUniPCMultistepScheduler`` updates are elementwise linear combinations of
        the sample and the model-output history, so splitting the flat latent into
        [non-action | action] segments and stepping them independently is exact.
        """
        assert len(noise) == len(seed) == len(action_spans), (
            f"noise({len(noise)}) / seed({len(seed)}) / action_spans({len(action_spans)}) length mismatch"
        )

        def _init(seed_i: int | None, shift_i: float) -> tuple[torch.Generator, FlowUniPCMultistepScheduler]:
            seed_g = torch.Generator(device=self.tensor_kwargs["device"])
            if seed_i is not None:
                seed_g.manual_seed(seed_i)
            sched = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.cfg.num_train_timesteps,
                shift=self.cfg.shift,
                use_dynamic_shifting=self.cfg.use_dynamic_shifting,
            )
            sched.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift_i)
            return (seed_g, sched)

        gens_vis, scheds_vis = zip(*[_init(s, shift) for s in seed])
        gens_act, scheds_act = zip(*[_init(s, shift_action) for s in seed])
        ts_vis = scheds_vis[0].timesteps
        ts_act = scheds_act[0].timesteps
        assert len(ts_vis) == len(ts_act), f"schedule length mismatch: {len(ts_vis)} vs {len(ts_act)}"
        latent = list(noise)
        for k in progress_bar(range(len(ts_vis)), desc="Sampling (dual-schedule)", total=len(ts_vis)):
            t_vis = ts_vis[k]
            t_act = ts_act[k]
            velocity_pred = velocity_fn(latent, t_vis.reshape(1, 1), timestep_action=t_act.reshape(1, 1))
            for i in range(len(latent)):
                x = latent[i]
                v = velocity_pred[i]
                span = action_spans[i]
                if span is None or span[1] == 0:
                    latent[i] = (
                        scheds_vis[i]
                        .step(
                            model_output=v,
                            timestep=t_vis,
                            sample=x.unsqueeze(0),
                            return_dict=False,
                            generator=gens_vis[i],
                        )[0]
                        .squeeze(0)
                    )
                    continue
                off, length = span
                x_vis = torch.cat([x[:off], x[off + length :]])
                v_vis = torch.cat([v[:off], v[off + length :]])
                new_vis = (
                    scheds_vis[i]
                    .step(
                        model_output=v_vis,
                        timestep=t_vis,
                        sample=x_vis.unsqueeze(0),
                        return_dict=False,
                        generator=gens_vis[i],
                    )[0]
                    .squeeze(0)
                )
                new_act = (
                    scheds_act[i]
                    .step(
                        model_output=v[off : off + length],
                        timestep=t_act,
                        sample=x[off : off + length].unsqueeze(0),
                        return_dict=False,
                        generator=gens_act[i],
                    )[0]
                    .squeeze(0)
                )
                latent[i] = torch.cat([new_vis[:off], new_act, new_vis[off:]])
        return latent
