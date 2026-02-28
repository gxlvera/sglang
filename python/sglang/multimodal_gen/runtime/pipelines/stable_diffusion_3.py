# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""StableDiffusion3 pipeline implementation."""

import torch

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    PipelineStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SD3ConditioningStage(PipelineStage):
    """Merge CLIP-T, CLIP-G and T5 embeddings into unified prompt/pooled tensors."""

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch = self._merge_embeddings(batch.prompt_embeds, batch.pooled_embeds, batch)
        if batch.do_classifier_free_guidance:
            batch = self._merge_negative_embeddings(
                batch.negative_prompt_embeds, batch.neg_pooled_embeds, batch
            )
        return batch

    def _merge_embeddings(
        self,
        prompt_embeds_list: list[torch.Tensor],
        pooled_embeds_list: list[torch.Tensor],
        batch: Req,
    ) -> Req:
        if len(prompt_embeds_list) != 3:
            raise ValueError(
                "SD3 requires exactly 3 prompt embedding tensors, "
                f"got {len(prompt_embeds_list)}."
            )
        if len(pooled_embeds_list) < 2:
            raise ValueError(
                "SD3 requires at least 2 pooled embedding tensors, "
                f"got {len(pooled_embeds_list)}."
            )

        clipt, clipg, t5 = prompt_embeds_list
        clip_merged = torch.cat([clipt, clipg], dim=-1)
        clip_merged = torch.nn.functional.pad(
            clip_merged, (0, t5.shape[-1] - clip_merged.shape[-1])
        )
        batch.prompt_embeds = [torch.cat([clip_merged, t5], dim=-2)]
        batch.pooled_embeds = [
            torch.cat([pooled_embeds_list[0], pooled_embeds_list[1]], dim=-1)
        ]
        return batch

    def _merge_negative_embeddings(
        self,
        neg_embeds_list: list[torch.Tensor],
        neg_pooled_list: list[torch.Tensor],
        batch: Req,
    ) -> Req:
        if len(neg_embeds_list) != 3:
            raise ValueError(
                "SD3 requires exactly 3 negative prompt embedding tensors, "
                f"got {len(neg_embeds_list)}."
            )
        if len(neg_pooled_list) < 2:
            raise ValueError(
                "SD3 requires at least 2 negative pooled embedding tensors, "
                f"got {len(neg_pooled_list)}."
            )

        neg_clipt, neg_clipg, neg_t5 = neg_embeds_list
        neg_clip_merged = torch.cat([neg_clipt, neg_clipg], dim=-1)
        neg_clip_merged = torch.nn.functional.pad(
            neg_clip_merged, (0, neg_t5.shape[-1] - neg_clip_merged.shape[-1])
        )
        batch.negative_prompt_embeds = [torch.cat([neg_clip_merged, neg_t5], dim=-2)]
        batch.neg_pooled_embeds = [
            torch.cat([neg_pooled_list[0], neg_pooled_list[1]], dim=-1)
        ]
        return batch


class StableDiffusion3Pipeline(ComposedPipelineBase):
    """StableDiffusion3 pipeline implementation."""

    pipeline_name = "StableDiffusion3Pipeline"

    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "tokenizer_2",
        "tokenizer_3",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())

        self.add_stage(
            TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                    self.get_module("text_encoder_3"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                    self.get_module("tokenizer_3"),
                ],
            ),
            "prompt_encoding_stage_primary",
        )

        self.add_stage(SD3ConditioningStage())

        self.add_standard_timestep_preparation_stage()
        self.add_standard_latent_preparation_stage()
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = StableDiffusion3Pipeline
