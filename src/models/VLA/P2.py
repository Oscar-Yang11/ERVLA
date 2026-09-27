import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from src.models import MIMODEL
from src.models.policy_head.DiT import DiT, TimestepEmbedder
from src.models.projector.linear_projector import MLPProjector
from src.models.VLM.qwen3vl import Qwen3VLForConditionalGeneration

from src.scheduler.flow import RectifiedFlow

from src.utils.model_utils import auto_cast

try:
    from src.utils.triton_unpad import unpad_all_triton
    _HAS_TRITON_UNPAD = True
except ImportError:
    _HAS_TRITON_UNPAD = False


@MIMODEL.register_module()
class P2(nn.Module):
    def __init__(
        self,
        state_shape=(1, 16),
        action_shape=(30, 29),
        dit_num_layers=16,
        dit_hidden_size=1024,
        dit_is_causal=True,
        num_steps=5,
        training_repeat=4,
        freeze_vlm=False,
        stop_gradient_to_vlm=False,
        enable_dct=False,
    ):
        super().__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape

        self.dit_num_layers = dit_num_layers
        self.dit_hidden_size = dit_hidden_size
        self.dit_is_causal = dit_is_causal

        self.num_steps = num_steps

        self.training_repeat = training_repeat

        self.freeze_vlm = freeze_vlm
        self.stop_gradient_to_vlm = stop_gradient_to_vlm
        self.dct_coefficient = 1. if enable_dct else 0.

        self._build_model()
        self._build_scheduler()

    def _build_model(self):
        # vlm
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct", attn_implementation="flash_attention_2", dtype=torch.bfloat16
        ).train()
        self.vlm.model.get_input_embeddings().requires_grad_(False)

        if self.freeze_vlm:
            self.vlm.requires_grad_(False)
        else:
            self.vlm.model.visual.gradient_checkpointing_enable()

        # dit
        self.dit = DiT(
            hidden_size=self.dit_hidden_size, kv_heads=8, layer_num=self.dit_num_layers, is_causal=self.dit_is_causal
        )

        # projector
        self.state_projector = MLPProjector(
            input_dim=self.state_shape[-1], output_dim=self.dit_hidden_size, num_layers=2
        )
        self.action_projector = MLPProjector(
            input_dim=self.action_shape[-1], output_dim=self.dit_hidden_size, num_layers=2
        )

        self.action_output_layer = MLPProjector(
            input_dim=self.dit_hidden_size,
            inter_dim=self.action_shape[-1],
            output_dim=self.action_shape[-1],
            num_layers=2,
        )

        # time embed
        self.t_embedder = TimestepEmbedder(self.dit_hidden_size)
        self.t_projector = MLPProjector(input_dim=self.dit_hidden_size, output_dim=6 * self.dit_hidden_size, bias=True)

        # rope
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(Qwen3VLTextConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct"))

        # sink token
        self.sink = nn.Embedding(1, self.dit_hidden_size)

        self.to(torch.bfloat16)

    def _build_scheduler(self):
        self.scheduler = RectifiedFlow()

    def dit_forward(self, noisy_action, t, action_mask, state_embed, position_embeds, past_key_values, attn_mask):
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
        hidden_states = self.dit(hidden_states, past_key_values, attn_mask, position_embeds, t_embeds)

        # action
        hidden_states = hidden_states[:, -noisy_action.shape[1] :, :]
        output = self.action_output_layer(hidden_states)

        return output
    
    @torch.no_grad()
    def generate(self, batch):
        return self.forward(batch, return_loss=False)


    @auto_cast
    def forward(self, batch, return_loss=False):
        action_segments = batch.pop("action_segments", None)

        # vlm forward
        vlm_outputs = self.vlm(**batch, use_cache=True)

        # get action
        action, action_mask, state = self.get_action_input(batch)

        action_bs, action_length, _ = action.shape
        _, state_length, _ = state.shape

        q_len = action_length + state_length + 1

        vlm_outputs.past_key_values, vlm_outputs.position_ids, vlm_outputs.attention_mask = self.unpad(
            vlm_outputs.past_key_values,
            vlm_outputs.position_ids,
            vlm_outputs.attention_mask,
            action_bs,
            action_segments,
        )

        ## position_embeds
        position_ids = (
            torch.arange(0, q_len, device=action.device).view(1, 1, -1).repeat(3, action_bs, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )
        position_ids = self.repeat(position_ids, dim=1)
        position_embeds = self.rotary_emb(action, position_ids)

        ## cache attention mask
        cache_mask = vlm_outputs.attention_mask
        cache_mask = cache_mask[:, None, :].expand(-1, q_len, -1)
        if self.dit_is_causal:
            causal_mask = torch.tril(torch.ones(action_bs, q_len, q_len, device=action.device), diagonal=0)
        else:
            causal_mask = torch.full((action_bs, q_len, q_len), True, device=action.device)
        attn_mask = torch.cat([cache_mask, causal_mask], dim=-1)[:, None]
        attn_mask = attn_mask.bool()
        attn_mask = self.repeat(attn_mask)

        ## state
        state_embed = self.state_projector(state)
        state_embed = self.repeat(state_embed)

        ## process action
        action = self.repeat(action)
        action_mask = self.repeat(action_mask)
        noise = torch.randn_like(action)

        if self.training:
            noisy_action, target, t = self.scheduler.training_data(x0=noise, x1=action)
            pred = self.dit_forward(
                noisy_action, t, action_mask, state_embed, position_embeds, vlm_outputs.past_key_values, attn_mask
            )
        else:
            target = action
            pred = self.scheduler.generate(
                noise,
                num_steps=self.num_steps,
                forward_func=self.dit_forward,
                action_mask=action_mask,
                state_embed=state_embed,
                position_embeds=position_embeds,
                past_key_values=vlm_outputs.past_key_values,
                attn_mask=attn_mask,
            )

        if return_loss:
            return self.compute_loss(pred, target, action_mask, vlm_outputs["loss"])
        else:
            return pred

    def get_action_input(self, batch):
        device = batch["input_ids"].device
        if "action" in batch:
            action = batch.pop("action")
            action_mask = batch.pop("action_mask", None)
            state = batch.pop("state")
        else:
            action = torch.zeros((1, *self.action_shape), device=device, dtype=torch.bfloat16)
            action_mask = torch.zeros_like(action, dtype=torch.int32)
            state = torch.zeros((1, *self.state_shape), device=device, dtype=torch.bfloat16)

        return action, action_mask, state

    def unpad(self, past_key_values, position_ids, attention_mask, action_bs, action_segments):
        if action_segments is None:
            return past_key_values, position_ids, attention_mask
        all_keys = torch.stack([layer[0] for layer in past_key_values], dim=0)
        all_values = torch.stack([layer[1] for layer in past_key_values], dim=0)

        if self.stop_gradient_to_vlm:
            all_keys = all_keys.detach()
            all_values = all_values.detach()

        seq_len = action_segments[:, 1] - action_segments[:, 0]

        assert len(seq_len) == action_bs
        max_length = seq_len.max()

        num_layers = all_keys.size(0)
        num_heads = all_keys.size(2)
        head_dim = all_keys.size(4)

        if _HAS_TRITON_UNPAD:
            starts = action_segments[:, 0]
            new_keys, new_values, new_position_ids = unpad_all_triton(
                all_keys, all_values, position_ids,
                starts, seq_len, action_bs, max_length,
            )
            arange = torch.arange(max_length, device=all_keys.device)
            new_attention_mask = (arange[None, :] < seq_len[:, None]).int()
        else:
            new_keys = torch.zeros(
                (num_layers, action_bs, num_heads, max_length, head_dim), device=all_keys.device, dtype=all_keys.dtype
            )
            new_values = torch.zeros_like(new_keys)
            new_position_ids = torch.zeros((3, action_bs, max_length), device=all_keys.device, dtype=position_ids.dtype)
            new_attention_mask = torch.zeros((action_bs, max_length), device=all_keys.device, dtype=torch.int)
            for i in range(action_bs):
                start = action_segments[i][0]
                end = action_segments[i][1]
                new_keys[:, i, :, : seq_len[i]] = all_keys[:, 0, :, start:end]
                new_values[:, i, :, : seq_len[i]] = all_values[:, 0, :, start:end]
                new_position_ids[:, i, : seq_len[i]] = position_ids[:, 0, start:end]
                new_attention_mask[i, : seq_len[i]] = 1

        new_past = list(zip(new_keys, new_values))

        return new_past, new_position_ids, new_attention_mask

    def repeat(self, x, dim=0):
        if not self.training:
            return x
        return x.repeat_interleave(self.training_repeat, dim=dim)

    def compute_loss(self, pred, target, action_mask, loss_vlm):
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
            loss_dct = loss_dct[batch_mask][action_mask_dct].mean()

        loss = 0.5 * loss_mse + self.dct_coefficient * loss_dct + 0.5 * loss_vlm

        return {"loss": loss, "loss_mse": loss_mse, "loss_dct": loss_dct, "loss_vlm": loss_vlm}
