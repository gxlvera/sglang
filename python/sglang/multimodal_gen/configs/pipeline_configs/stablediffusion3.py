# SPDX-License-Identifier: Apache-2.0
"""Stable Diffusion 3 pipeline configuration."""

import os
from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import StableDiffusion3TransformerConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfigForSD3,
    T5ConfigForSD3,
)
from sglang.multimodal_gen.configs.models.vaes.stablediffusion3 import (
    StableDiffusion3VAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
)


def t5_preprocess_text(prompt: str) -> str:
    return prompt


def clip_preprocess_text(prompt: str) -> str:
    return prompt


def clip_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    """
    Keep raw hidden states and select pre-final layer in TextEncodingStage
    to match sglang-2 SD3 implementation.
    """
    assert outputs.hidden_states is not None
    return outputs.hidden_states


def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    return outputs.last_hidden_state


@dataclass
class StableDiffusion3PipelineConfig(SpatialImagePipelineConfig):
    """
    Configuration for SD3 text/image generation.

    This config intentionally relies on SD3-specific encoder configs to provide
    tokenizer kwargs, instead of stage-level tokenizer overrides.
    """

    task_type: ModelTaskType = ModelTaskType.T2I

    # Model configurations
    dit_config: DiTConfig = field(default_factory=StableDiffusion3TransformerConfig)
    vae_config: VAEConfig = field(default_factory=StableDiffusion3VAEConfig)

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (
            CLIPTextConfigForSD3(),
            CLIPTextConfigForSD3(),
            T5ConfigForSD3(),
        )
    )

    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("fp16", "fp16", "fp32")
    )

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (
            clip_preprocess_text,
            clip_preprocess_text,
            t5_preprocess_text,
        )
    )

    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput, dict], torch.Tensor], ...] = (
        field(
            default_factory=lambda: (
                clip_postprocess_text,
                clip_postprocess_text,
                t5_postprocess_text,
            )
        )
    )

    # SD3 specific parameters
    guidance_scale: float = 7.0
    use_precision_specific_weights = True
    vae_model_name = "diffusion_pytorch_model"

    def __post_init__(self):
        self.dit_config.update_model_arch({"_class_name": "SD3Transformer2DModel"})

        configs = list(self.text_encoder_configs)
        configs[0].update_model_arch({"_class_name": "CLIPTextModelWithProjection"})
        configs[1].update_model_arch({"_class_name": "CLIPTextModelWithProjection"})
        configs[2].update_model_arch({"_class_name": "T5EncoderModel"})
        self.text_encoder_configs = tuple(configs)

    def _maybe_debug_encoder_hidden_states(
        self, tensor: torch.Tensor, tag: str = "pos"
    ) -> None:
        if os.getenv("SGLANG_DEBUG_ENCODER_HIDDEN_STATES", "0") != "1":
            return

        max_elems = int(os.getenv("SGLANG_DEBUG_MAX_ELEMS", "64"))
        t_cpu = tensor.detach().float().cpu()
        flat = t_cpu.reshape(-1)
        preview = flat[:max_elems].tolist()
        print(
            f"[SD3 DEBUG][{tag}] encoder_hidden_states "
            f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
        )
        print(
            f"[SD3 DEBUG][{tag}] encoder_hidden_states "
            f"first_{max_elems}={preview}"
        )

        if os.getenv("SGLANG_DEBUG_FULL_ENCODER_HIDDEN_STATES", "0") == "1":
            torch.set_printoptions(profile="full")
            print(f"[SD3 DEBUG][{tag}] encoder_hidden_states_full=\n{t_cpu}")

        dump_path = os.getenv("SGLANG_DEBUG_DUMP_PATH")
        if dump_path:
            payload = {
                "tag": tag,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor": t_cpu,
            }
            torch.save(payload, dump_path)
            print(f"[SD3 DEBUG][{tag}] saved tensor to {dump_path}")

    def get_pos_prompt_embeds(self, batch):
        tensor = batch.prompt_embeds[0]
        self._maybe_debug_encoder_hidden_states(tensor, tag="pos")
        return tensor

    def get_neg_prompt_embeds(self, batch):
        tensor = batch.negative_prompt_embeds[0]
        self._maybe_debug_encoder_hidden_states(tensor, tag="neg")
        return tensor

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "pooled_projections": (
                batch.pooled_embeds[0] if batch.pooled_embeds else None
            )
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "pooled_projections": (
                batch.neg_pooled_embeds[0] if batch.neg_pooled_embeds else None
            )
        }

    # SD3 image latents are spatial (B, C, H, W), not video-like (B, C, T, H, W).
    def prepare_latent_shape(self, batch, batch_size, num_frames):  # noqa: ARG002
        spatial_ratio = getattr(
            self.vae_config.arch_config, "spatial_compression_ratio", 8
        )
        in_channels = getattr(self.dit_config.arch_config, "in_channels", 16)
        return (
            batch_size,
            in_channels,
            int(batch.height) // spatial_ratio,
            int(batch.width) // spatial_ratio,
        )
