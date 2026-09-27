import torch
import torch.nn as nn
import torch.nn.functional as F

# patch
from src.models.VLM import qwen3vl_expert

from src.models.VLM.qwen3vl_lce import lce_forward as qwen3_vl_lce_forward
from liger_kernel.transformers.rope import liger_rotary_pos_emb, liger_rotary_pos_emb_vision
from liger_kernel.transformers.rms_norm import LigerRMSNorm

qwen3vl_expert.apply_rotary_pos_emb = liger_rotary_pos_emb
# qwen3vl_expert.apply_rotary_pos_emb_vision = liger_rotary_pos_emb_vision
qwen3vl_expert.Qwen3VLTextRMSNorm = LigerRMSNorm
qwen3vl_expert.Qwen3VLForConditionalGeneration.forward = qwen3_vl_lce_forward
# patch end

from transformers import AutoProcessor
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from src.models import MIMODEL
from src.models.projector.linear_projector import MLPProjector
from src.models.VLM.qwen3vl_expert import Qwen3VLForConditionalGeneration, STATE_ID

from src.utils.model_utils import auto_cast


@MIMODEL.register_module()
class P2_L1(nn.Module):
    def __init__(
        self,
        state_shape=(1, 16),
        action_shape=(30, 29)
    ):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape

        self._build_model()

    def _build_model(self):
        # vlm
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", attn_implementation="flash_attention_2", dtype=torch.bfloat16
        ).train()
        self.vlm.model.get_input_embeddings().requires_grad_(False)
        self.vlm.model.visual.gradient_checkpointing_enable()
        self.vlm_hidden_size = self.vlm.config.text_config.hidden_size

        # state
        self.state_projector_vlm = MLPProjector(
            input_dim=self.state_shape[-1], output_dim=self.vlm_hidden_size, num_layers=2
        )

        # action
        self.action_projector_vlm = nn.Sequential(
            MLPProjector(input_dim=self.vlm_hidden_size, output_dim=self.vlm_hidden_size, num_layers=4),
            MLPProjector(input_dim=self.vlm_hidden_size, output_dim=self.action_shape[-1], num_layers=1)
        )

        self.to(torch.bfloat16)

    @auto_cast
    def forward(self, batch, return_loss=False):
        batch.pop("action", None)
        batch.pop("action_mask", None)

        vlm_state = batch.pop("state", None)
        if vlm_state == None:
            vlm_state = torch.zeros((1, *self.state_shape), dtype=torch.bfloat16, device=batch["input_ids"].device)
        vlm_state = vlm_state.flatten(0, 1)
        vlm_action_target, vlm_action_mask = batch.pop("vlm_action_target", None), batch.pop("vlm_action_mask", None)

        state_embeds = self.state_projector_vlm(vlm_state)
        batch["state_embeds"] = state_embeds

        vlm_outputs = self.vlm(**batch, use_cache=False)

        action_index = batch["input_ids"] > STATE_ID
        if not action_index.any():
            action_hidden = vlm_outputs["hidden_states"][0, 0:1]
        else:
            action_hidden = vlm_outputs["hidden_states"][action_index]

        action_pred = self.action_projector_vlm(action_hidden)

        if return_loss:
            if not action_index.any():
                loss_l1 = (action_pred[0, 0] - state_embeds[0, 0]) * 0.0
            else:
                loss_l1 = F.l1_loss(action_pred, vlm_action_target, reduction="none")
                loss_l1 = loss_l1[vlm_action_mask.bool()].mean()
        
            loss = 0.5 * vlm_outputs["loss"] + 0.5 * loss_l1
            return {"loss": loss, "loss_vlm": vlm_outputs["loss"], "loss_l1": loss_l1}

        else:
            return action_pred
    
    @auto_cast
    @torch.no_grad()
    def generate(self, batch):
        return self(batch, return_loss=False)
