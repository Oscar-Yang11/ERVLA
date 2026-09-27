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
from src.models.VLM.qwen3vl_expert import Qwen3VLForConditionalGeneration, STATE_ID, SCORE_ID

from src.utils.model_utils import auto_cast


@MIMODEL.register_module()
class P2_Choice(nn.Module):
    def __init__(
        self,
        state_shape=(1, 16),
        action_shape=(30, 29),
        n_choices=5,
        soft_routing=False,
        softmax_temperature=1.0,
    ):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.n_choices = n_choices
        self.soft_routing = soft_routing
        self.softmax_temperature = softmax_temperature

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
            MLPProjector(input_dim=self.vlm_hidden_size, output_dim=self.action_shape[-1] * self.n_choices, num_layers=1)
        )

        # score
        self.score_projector_vlm = nn.Sequential(
            MLPProjector(input_dim=self.vlm_hidden_size, output_dim=self.vlm_hidden_size, num_layers=4),
            MLPProjector(input_dim=self.vlm_hidden_size, output_dim=self.n_choices, num_layers=1)
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
        score_index = batch["input_ids"] == SCORE_ID
        if not action_index.any():
            action_hidden = vlm_outputs["hidden_states"][0, 0:1]
            score_hidden = vlm_outputs["hidden_states"][0, 0:1]
        else:
            action_hidden = vlm_outputs["hidden_states"][action_index]
            score_hidden = vlm_outputs["hidden_states"][score_index]

        action_pred = self.action_projector_vlm(action_hidden) # (b * t, n * d)
        score_pred = self.score_projector_vlm(score_hidden) # (b, n)

        if return_loss:
            if not action_index.any():
                loss_action = (action_pred[0, 0] - state_embeds[0, 0]) * 0.0
                loss_score = (action_pred[0, 0] - state_embeds[0, 0]) * 0.0
            else:
                # get action actual length
                action_actual_length = batch['vlm_action_actual_length']
                assert action_actual_length.sum() == action_pred.shape[0]
                action_actual_length = action_actual_length.detach().tolist()
                assert len(action_actual_length) == score_pred.shape[0]
                start = 0
                loss_score_list = []
                loss_action_list = []

                for batch_i, sample_action_length in enumerate(action_actual_length):
                    end = start + sample_action_length
                    sample_action_pred = action_pred[start:end] # (t, n * d)
                    sample_action_target = vlm_action_target[start:end] # (t, d)
                    sample_action_mask = vlm_action_mask[start:end] # (t, d)
                    sample_score_pred = score_pred[batch_i] # (n, )

                    # reshape action prediction
                    sample_action_pred = sample_action_pred.reshape(sample_action_length, self.n_choices, -1) # (t, n, d)
                    sample_action_pred = sample_action_pred.transpose(1, 0) # (n, t, d)

                    # reshape action mask
                    sample_action_mask_rep = sample_action_mask[None, ...] # (1, t, d)
                    sample_action_mask_rep = sample_action_mask_rep.repeat(self.n_choices, 1, 1) # (n, t, d)

                    # reshape action target
                    sample_action_target_rep = sample_action_target[None, ...] # (1, t, d)
                    sample_action_target_rep = sample_action_target_rep.repeat(self.n_choices, 1, 1) # (n, t, d)

                    # compute difference between prediction & target
                    sample_action_abs_diff = F.l1_loss(sample_action_pred, sample_action_target_rep, reduction="none") # (n, t, d)

                    # compute ground-truth for score
                    sample_action_abs_diff_masked = sample_action_abs_diff[sample_action_mask_rep.bool()].reshape(self.n_choices, -1) # (n, -1)
                    sample_score_gt = sample_action_abs_diff_masked.mean(dim=-1) # (n, )

                    # compute score loss
                    sample_loss_score = ((sample_score_pred - sample_score_gt.detach()) ** 2).mean()

                    if self.soft_routing and self.training:
                        # compute loss for the action with softmax score
                        sample_loss_score_softmax = torch.softmax(-sample_score_gt / self.softmax_temperature, dim=0) # (n, )
                        sample_loss_action = sample_score_gt * sample_loss_score_softmax # (n, )
                        sample_loss_action = sample_loss_action.sum()
                    else:
                        # compute loss for the action with the best score
                        choice_idx = sample_score_gt.argmin()
                        sample_loss_action = sample_score_gt[choice_idx]

                    # insert to list
                    loss_score_list.append(sample_loss_score)
                    loss_action_list.append(sample_loss_action)

                    # update start
                    start = end
                
                assert end == action_pred.shape[0]
                loss_action = torch.stack(loss_action_list).mean()
                loss_score = torch.stack(loss_score_list).mean()
        
            loss = 0.5 * vlm_outputs["loss"] + 0.5 * loss_action + 0.5 * loss_score
            return {"loss": loss, "loss_vlm": vlm_outputs["loss"], "loss_l1": loss_action, "loss_score": loss_score}

        else:
            raise NotImplementedError
    
    @auto_cast
    @torch.no_grad()
    def generate(self, batch):
        return self(batch, return_loss=False)
