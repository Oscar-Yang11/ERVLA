import random

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

from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from src.models import MIMODEL
from src.models.policy_head.DiT import DiT, TimestepEmbedder, ConcatDiT
from src.models.projector.linear_projector import MLPProjector
from src.models.VLM.qwen3vl_expert import Qwen3VLForConditionalGeneration, STATE_ID, SCORE_ID
from src.models.VLA.P2 import P2
from src.models.VLA.P2_Local import P2_Local

from src.utils.model_utils import auto_cast


@MIMODEL.register_module()
class P2_Choice_KI(P2):
    def __init__(
        self,
        state_shape=(1, 16),
        action_shape=(30, 29),
        n_choices=5,
        dit_num_layers=36,
        dit_hidden_size=1024,
        dit_is_causal=True,
        num_steps=5,
        training_repeat=4,
        freeze_vlm=False,
        stop_gradient_to_vlm=False,
        enable_dct=False,
        knowledge_insulation=True,
    ):
        self.n_choices = n_choices
        self.knowledge_insulation = knowledge_insulation
        super().__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            dit_num_layers=dit_num_layers,
            dit_hidden_size=dit_hidden_size,
            dit_is_causal=dit_is_causal,
            num_steps=num_steps,
            training_repeat=training_repeat,
            freeze_vlm=freeze_vlm,
            stop_gradient_to_vlm=stop_gradient_to_vlm,
            enable_dct=enable_dct,
        )

    def _build_model(self):
        # vlm
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", 
            attn_implementation="flash_attention_2", 
            dtype=torch.bfloat16
        ).train()

        self.vlm.model.get_input_embeddings().requires_grad_(False)

        if self.freeze_vlm:
            self.vlm.requires_grad_(False)
        else:
            self.vlm.model.visual.gradient_checkpointing_enable()
        
        self.vlm_hidden_size = self.vlm.config.text_config.hidden_size

        # projector for choice
        self.state_projector_choice = MLPProjector(
            input_dim=self.state_shape[-1], 
            inter_dim=self.vlm_hidden_size,
            output_dim=self.vlm_hidden_size, 
            num_layers=2
        )
        self.action_projector_choice = nn.Sequential(
            MLPProjector(
                input_dim=self.vlm_hidden_size, 
                inter_dim=self.vlm_hidden_size,
                output_dim=self.vlm_hidden_size, 
                num_layers=4
            ),
            MLPProjector(
                input_dim=self.vlm_hidden_size, 
                output_dim=self.action_shape[-1] * self.n_choices, 
                num_layers=1
            )
        )
        self.score_projector_choice = nn.Sequential(
            MLPProjector(
                input_dim=self.vlm_hidden_size, 
                inter_dim=self.vlm_hidden_size,
                output_dim=self.vlm_hidden_size, 
                num_layers=4
            ),
            MLPProjector(
                input_dim=self.vlm_hidden_size, 
                output_dim=self.n_choices, 
                num_layers=1
            )
        )

        # dit
        self.dit = DiT(
            hidden_size=self.dit_hidden_size, 
            kv_heads=8, 
            layer_num=self.dit_num_layers, 
            is_causal=self.dit_is_causal
        )

        # projector for dit
        self.state_projector = MLPProjector(
            input_dim=self.state_shape[-1], 
            inter_dim=self.dit_hidden_size,
            output_dim=self.dit_hidden_size, 
            num_layers=2
        )
        self.action_projector = MLPProjector(
            input_dim=self.action_shape[-1], 
            inter_dim=self.dit_hidden_size,
            output_dim=self.dit_hidden_size, 
            num_layers=2
        )
        self.action_output_layer = MLPProjector(
            input_dim=self.dit_hidden_size, 
            # Match the released MiBoT HF checkpoint: the output head uses
            # action_dim as its intermediate width.
            inter_dim=self.action_shape[-1],
            output_dim=self.action_shape[-1], 
            num_layers=2
        )

        # time embed
        self.t_embedder = TimestepEmbedder(self.dit_hidden_size)
        self.t_projector = MLPProjector(
            input_dim=self.dit_hidden_size, 
            output_dim=6 * self.dit_hidden_size, 
            bias=True
        )

        # rope
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(
            Qwen3VLTextConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        )

        # sink token
        self.sink = nn.Embedding(1, self.dit_hidden_size)

        self.to(torch.bfloat16)

    def get_action_vlm_condition_segments(self, batch):
        if self.knowledge_insulation:
            return batch.pop("action_vlm_condition_segments", None)
        else:
            return batch.pop("action_segments", None)
        
    # @torch.compile(dynamic=True)
    def dit_forward(self, noisy_action, t, action_mask, state_embed, position_embeds, past_key_values, perf_attn_mask, repeat=1):
        # time step
        t_embeds = self.t_embedder(t[:, 0, 0] * 1000)
        t_embeds = self.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)

        # action projection
        noisy_action = noisy_action * action_mask
        noisy_action = self.action_projector(noisy_action)

        # concat
        sink = self.sink.weight[None].repeat(state_embed.shape[0], 1, 1)
        hidden_states = torch.cat([sink, state_embed, noisy_action], dim=1).contiguous()

        # dit
        hidden_states = self.dit.forward_perf(hidden_states, past_key_values, perf_attn_mask, position_embeds, t_embeds, repeat)

        # action
        hidden_states = hidden_states[:, -noisy_action.shape[1] :, :]
        output = self.action_output_layer(hidden_states)

        return output

    @auto_cast
    def forward(self, batch, return_loss=False):
        action_vlm_condition_segments = self.get_action_vlm_condition_segments(batch)
        vlm_action_actual_length = batch.pop("vlm_action_actual_length", None)
        
        action, action_mask, state = self.get_action_input(batch)

        vlm_state = state.flatten(0, 1)
        vlm_action_target, vlm_action_mask = batch.pop("vlm_action_target", None), batch.pop("vlm_action_mask", None)

        vlm_state_embeds = self.state_projector_choice(vlm_state)
        batch["state_embeds"] = vlm_state_embeds

        vlm_outputs = self.vlm(**batch, use_cache=True)

        ##### Predict action with Choice #####
        action_index = batch["input_ids"] > STATE_ID
        score_index = batch["input_ids"] == SCORE_ID
        if not action_index.any():
            action_hidden = vlm_outputs["hidden_states"][0, 0:1]
            score_hidden = vlm_outputs["hidden_states"][0, 0:1]
        else:
            action_hidden = vlm_outputs["hidden_states"][action_index]
            score_hidden = vlm_outputs["hidden_states"][score_index]
        action_pred_choice = self.action_projector_choice(action_hidden) # (b * t, n * d)
        score_pred_choice = self.score_projector_choice(score_hidden) # (b, n)

        ##### Predict action with DiT #####
        action_bs, action_length, _ = action.shape
        _, state_length, _ = state.shape
        
        q_len = action_length + state_length + 1

        vlm_outputs.past_key_values, vlm_outputs.position_ids, vlm_outputs.attention_mask = self.unpad(
            vlm_outputs.past_key_values,
            vlm_outputs.position_ids,
            vlm_outputs.attention_mask,
            action_bs,
            action_vlm_condition_segments,
        )

        ## position_embeds
        position_ids = (
            torch.arange(0, q_len, device=action.device).view(1, 1, -1).repeat(3, action_bs, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )
        repeated_position_ids = self.repeat(position_ids, dim=1)
        repeated_position_embeds = self.rotary_emb(action, repeated_position_ids)

        # create mask for DiT
        if self.dit_is_causal:
            base_causal_mask = torch.tril(torch.ones(action_bs, q_len, q_len, device=action.device), diagonal=0)
        else:
            base_causal_mask = torch.full((action_bs, q_len, q_len), True, device=action.device)
        repeat = self.training_repeat if self.training else 1
        causal_mask = torch.zeros(action_bs, repeat * q_len, repeat * q_len).to(base_causal_mask)
        for k in range(repeat):
            s = k * q_len
            e = (k + 1) * q_len
            causal_mask[:, s:e, s:e] = base_causal_mask
        cache_mask = vlm_outputs.attention_mask
        cache_mask = cache_mask[:, None, :].expand(-1, repeat * q_len, -1)
        perf_attn_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None]
        perf_attn_mask = perf_attn_mask.bool()

        ## state
        state_embed = self.state_projector(state)
        repeated_state_embed = self.repeat(state_embed)

        ## process action
        repeated_action = self.repeat(action)
        repeated_action_mask = self.repeat(action_mask)
        noise = torch.randn_like(repeated_action)

        if self.training:
            noisy_action, target, t = self.scheduler.training_data(x0=noise, x1=repeated_action)
            pred = self.dit_forward(
                noisy_action, t, repeated_action_mask, repeated_state_embed, repeated_position_embeds, vlm_outputs.past_key_values, perf_attn_mask=perf_attn_mask, repeat=self.training_repeat
            )
        else:
            target = repeated_action
            pred = self.scheduler.generate(
                noise,
                num_steps=self.num_steps,
                forward_func=self.dit_forward,
                action_mask=repeated_action_mask,
                state_embed=repeated_state_embed,
                position_embeds=repeated_position_embeds,
                past_key_values=vlm_outputs.past_key_values,
                perf_attn_mask=perf_attn_mask,
            )

        if return_loss:
            loss_vlm = vlm_outputs["loss"]
            loss_mse, loss_dct = self.compute_flow_loss(pred, target, repeated_action_mask)

            if vlm_action_target != None:
                loss_choice_l1, loss_choice_score = self.compute_choice_loss(
                    action_pred_choice,
                    score_pred_choice, 
                    vlm_action_target,
                    vlm_action_mask,
                    vlm_action_actual_length,
                )
            else:
                zero = action_pred_choice.sum() * 0.0 + score_pred_choice.sum() * 0.0
                loss_choice_l1 = zero
                loss_choice_score = zero

            loss = 0.5 * loss_vlm + 0.5 * loss_mse + self.dct_coefficient * loss_dct + 0.5 * loss_choice_l1 + 0.5 * loss_choice_score
            
            return {
                "loss": loss,
                "loss_vlm": loss_vlm,
                "loss_mse": loss_mse,
                "loss_dct": loss_dct,
                "loss_l1": loss_choice_l1,
                "loss_score": loss_choice_score,
            }
        else:
            return pred

    def compute_flow_loss(self, pred, target, action_mask):
        pred = pred.float()
        target = target.float()

        if action_mask.size(0) == 1 and (not torch.any(action_mask)):
            loss_mse = (pred[0, 0, -1] - target[0, 0, -1]) * 0.0
            loss_dct = loss_mse
        else:
            action_mask = action_mask.bool()
            
            # loss_mse
            loss_mse_unreduced = F.mse_loss(pred, target, reduction="none")
            loss_mse = loss_mse_unreduced[action_mask].mean()

            # loss_dct
            loss_dct = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
            batch_mask = action_mask[:, -1].any(dim=1)
            action_mask_dct = action_mask[batch_mask][:, : loss_dct.shape[1]]
            loss_dct = loss_dct[batch_mask][action_mask_dct]
            loss_dct = loss_dct.sum() / max(loss_dct.numel(), 1)

        return loss_mse, loss_dct

    def compute_choice_loss(self, action_pred, score_pred, action_target, action_mask, action_actual_length):
        # get action actual length
        assert action_actual_length.sum() == action_pred.shape[0]
        action_actual_length = action_actual_length.detach().tolist()
        assert len(action_actual_length) == score_pred.shape[0]
        start = 0
        loss_score_list = []
        loss_action_list = []

        for batch_i, sample_action_length in enumerate(action_actual_length):
            end = start + sample_action_length
            sample_action_pred = action_pred[start:end] # (t, n * d)
            sample_action_target = action_target[start:end] # (t, d)
            sample_action_mask = action_mask[start:end] # (t, d)
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

            # compute loss for the action with the best score
            choice_idx = sample_score_gt.argmin()
            sample_loss_action = sample_score_gt[choice_idx]

            # insert to list
            loss_score_list.append(sample_loss_score)
            loss_action_list.append(sample_loss_action)

            # update start
            start = end
        
        assert end == action_pred.shape[0]
        loss_choice_l1 = torch.stack(loss_action_list).mean()
        loss_choice_score = torch.stack(loss_score_list).mean()
        
        return loss_choice_l1, loss_choice_score


@MIMODEL.register_module()
class P2_Local_Choice_KI(P2_Choice_KI):
    def __init__(
        self,
        state_shape=(1, 16),
        action_shape=(30, 29),
        n_choices=5,
        dit_num_layers=36,
        dit_hidden_size=1024,
        dit_is_causal=True,
        num_steps=5,
        training_repeat=4,
        freeze_vlm=False,
        stop_gradient_to_vlm=False,
        enable_dct=False,
        knowledge_insulation=True,
        prefix_mask_prob=0.5,
        activate_async=0,
    ):
        super().__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            n_choices=n_choices,
            dit_num_layers=dit_num_layers,
            dit_hidden_size=dit_hidden_size,
            dit_is_causal=dit_is_causal,
            num_steps=num_steps,
            training_repeat=training_repeat,
            freeze_vlm=freeze_vlm,
            stop_gradient_to_vlm=stop_gradient_to_vlm,
            enable_dct=enable_dct,
            knowledge_insulation=knowledge_insulation,
        )
        self.prefix_mask_prob = prefix_mask_prob
        self.activate_async = activate_async

    def dit_forward(self, noisy_action, t, action_mask, state_embed, position_embeds, past_key_values, attn_mask, prefix_length):
        return P2_Local.dit_forward(self, noisy_action, t, action_mask, state_embed, position_embeds, past_key_values, attn_mask, prefix_length)
    
    @auto_cast
    def forward(self, batch, return_loss=False):
        action_vlm_condition_segments = self.get_action_vlm_condition_segments(batch)
        vlm_action_actual_length = batch.pop("vlm_action_actual_length", None)
        
        action, action_mask, state = self.get_action_input(batch)

        vlm_state = state.flatten(0, 1)
        vlm_action_target, vlm_action_mask = batch.pop("vlm_action_target", None), batch.pop("vlm_action_mask", None)

        vlm_state_embeds = self.state_projector_choice(vlm_state)
        batch["state_embeds"] = vlm_state_embeds

        vlm_outputs = self.vlm(**batch, use_cache=True)

        ##### Predict action with Choice #####
        action_index = batch["input_ids"] > STATE_ID
        score_index = batch["input_ids"] == SCORE_ID
        if not action_index.any():
            action_hidden = vlm_outputs["hidden_states"][0, 0:1]
            score_hidden = vlm_outputs["hidden_states"][0, 0:1]
        else:
            action_hidden = vlm_outputs["hidden_states"][action_index]
            score_hidden = vlm_outputs["hidden_states"][score_index]
        action_pred_choice = self.action_projector_choice(action_hidden) # (b * t, n * d)
        score_pred_choice = self.score_projector_choice(score_hidden) # (b, n)

        ##### Predict action with DiT #####
        action_bs, action_length, _ = action.shape
        _, state_length, _ = state.shape
        
        q_len = action_length + state_length + 1

        vlm_outputs.past_key_values, vlm_outputs.position_ids, vlm_outputs.attention_mask = self.unpad(
            vlm_outputs.past_key_values,
            vlm_outputs.position_ids,
            vlm_outputs.attention_mask,
            action_bs,
            action_vlm_condition_segments,
        )

        ## position_embeds
        if self.training:
            prefix_length = 0
            if getattr(self, "_global_step", 0) >= self.activate_async and random.random() < 0.5:
                prefix_length = random.randint(1, 6)
        else:
            prefix_length = batch.get("prefix_length", 0)
        prefix = action[:, :prefix_length]

        position_ids = (
            torch.arange(0, q_len, device=action.device).view(1, 1, -1).repeat(3, action_bs, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )
        position_ids[:, :, -(action_length - prefix_length):] += 10

        repeated_position_ids = self.repeat(position_ids, dim=1)
        repeated_position_embeds = self.rotary_emb(action, repeated_position_ids)

        ## cache attention mask
        cache_mask = vlm_outputs.attention_mask
        cache_mask = cache_mask[:, None, :].expand(-1, q_len, -1)
        if self.dit_is_causal:
            causal_mask = torch.tril(torch.ones(action_bs, q_len, q_len, device=action.device), diagonal=0)
            if self.training and prefix_length > 2:
                causal_mask = P2_Local._random_mask_prefix(self, causal_mask, prefix_length, state_length)
        else:
            causal_mask = torch.full((action_bs, q_len, q_len), True, device=action.device)
        attn_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None]
        attn_mask = attn_mask.bool()
        repeated_attn_mask = self.repeat(attn_mask)

        ## state
        state_embed = self.state_projector(state)
        repeated_state_embed = self.repeat(state_embed)

        ## process action
        repeated_action = self.repeat(action)
        repeated_prefix = self.repeat(prefix)
        repeated_action_mask = self.repeat(action_mask)
        noise = torch.randn_like(repeated_action)

        if self.training:
            noisy_action, target, t = self.scheduler.training_data(x0=noise, x1=repeated_action)
            pred = self.dit_forward(
                torch.cat([repeated_prefix, noisy_action[:, prefix_length:]], dim=1),
                t,
                repeated_action_mask,
                repeated_state_embed,
                repeated_position_embeds,
                vlm_outputs.past_key_values,
                repeated_attn_mask,
                prefix_length
            )
            pred = pred[:, prefix_length:]
            target = target[:, prefix_length:]

            if prefix_length > 0:
                with torch.no_grad():
                    pred_prefix = self.scheduler.generate(
                        torch.cat([repeated_prefix, noise[:, prefix_length:]], dim=1),
                        num_steps=self.num_steps,
                        forward_func=self.dit_forward,
                        action_mask=repeated_action_mask,
                        state_embed=repeated_state_embed,
                        position_embeds=repeated_position_embeds,
                        past_key_values=vlm_outputs.past_key_values,
                        attn_mask=repeated_attn_mask,
                        prefix_length=prefix_length,
                    )
                weight = (pred_prefix[:, prefix_length:] - repeated_action[:, prefix_length:]).abs()
            else:
                weight = torch.ones_like(pred)
            repeated_action_mask = repeated_action_mask[:, prefix_length:]
        else:
            target = repeated_action
            pred = self.scheduler.generate(
                torch.cat([repeated_prefix, noise[:, prefix_length:]], dim=1),
                num_steps=self.num_steps,
                forward_func=self.dit_forward,
                action_mask=repeated_action_mask,
                state_embed=repeated_state_embed,
                position_embeds=repeated_position_embeds,
                past_key_values=vlm_outputs.past_key_values,
                attn_mask=repeated_attn_mask,
                prefix_length=prefix_length,
            )
            weight = torch.ones_like(pred)

        if return_loss:
            loss_vlm = vlm_outputs["loss"]
            loss_mse, loss_dct = self.compute_flow_loss(pred, target, repeated_action_mask, weight)

            if vlm_action_target != None:
                loss_choice_l1, loss_choice_score = self.compute_choice_loss(
                    action_pred_choice,
                    score_pred_choice, 
                    vlm_action_target,
                    vlm_action_mask,
                    vlm_action_actual_length,
                )
            else:
                zero = action_pred_choice.sum() * 0.0 + score_pred_choice.sum() * 0.0
                loss_choice_l1 = zero
                loss_choice_score = zero

            loss = 0.5 * loss_vlm + 0.5 * loss_mse + self.dct_coefficient * loss_dct + 0.5 * loss_choice_l1 + 0.5 * loss_choice_score
            
            return {
                "loss": loss,
                "loss_vlm": loss_vlm,
                "loss_mse": loss_mse,
                "loss_dct": loss_dct,
                "loss_l1": loss_choice_l1,
                "loss_score": loss_choice_score,
            }
        else:
            return pred

    def compute_flow_loss(self, pred, target, action_mask, weight):
        pred = pred.float()
        target = target.float()
        weight = weight.float()

        if action_mask.size(0) == 1 and (not torch.any(action_mask)):
            loss_mse = (pred[0, 0, -1] - target[0, 0, -1]) * 0.0
            loss_dct = loss_mse
        else:
            action_mask = action_mask.bool()
            
            # loss_mse
            with torch.no_grad():
                weight[action_mask] = weight[action_mask] / weight[action_mask].mean()
                weight = torch.clamp(weight, min=0.5, max=5)
            loss_mse_unreduced = F.mse_loss(pred, target, reduction="none") * weight
            loss_mse = loss_mse_unreduced[action_mask].mean()

            # loss_dct
            loss_dct = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
            batch_mask = action_mask[:, -1].any(dim=1)
            weight_dct = weight.mean(dim=[1, 2])
            action_mask_dct = action_mask[batch_mask][:, : loss_dct.shape[1]]
            loss_dct = loss_dct * weight_dct.unsqueeze(1).unsqueeze(2)
            loss_dct = loss_dct[batch_mask][action_mask_dct].mean()

        return loss_mse, loss_dct
