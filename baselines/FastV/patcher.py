import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from .config import FastVConfig
from .ranking import AttentionScoreRanker


class FastVPatcher:
    """
    Implements FastV (Chen et al., 2024): after layer K, rank image tokens by
    the attention score they received in layer K and keep only the top
    (1 - R%). Text tokens are never pruned. Layers after K then run on the
    shorter sequence, so their self-attention and FFN costs drop accordingly.
    """

    def __init__(self, config: FastVConfig):
        self.config = config
        self.ranker = AttentionScoreRanker()
        self._original_forward = None
        self._pending_image_mask = None

    def patch_model(self, model: torch.nn.Module):
        language_model = self._get_language_model(model)
        self._original_forward = language_model.forward
        model.register_forward_pre_hook(self._create_input_hook(), with_kwargs=True)
        language_model.forward = self._create_patched_forward(language_model)

    def _get_language_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        raise AttributeError("Unsupported model structure: expected model.model.language_model.")

    def _create_input_hook(self):
        def hook(module, args, kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            if input_ids is not None:
                self._pending_image_mask = (input_ids == self.config.image_token_index)
            else:
                self._pending_image_mask = None
        return hook

    def _create_patched_forward(self, language_model: torch.nn.Module):
        patcher = self

        def patched_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            **kwargs,
        ):
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if position_ids is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
                position_ids = position_ids.unsqueeze(0)

            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

            is_prefill = hidden_states.shape[1] > 1
            image_mask = patcher._pending_image_mask
            patcher._pending_image_mask = None
            should_prune = (
                is_prefill
                and image_mask is not None
                and image_mask.any()
                and patcher.config.r > 0.0
            )

            captured_attn_weights = {}

            def capture_attn_hook(module, args, kwargs, output):
                captured_attn_weights["weights"] = output[1]

            for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                handle = None
                if should_prune and layer_idx == patcher.config.k:
                    prev_impl = self.config._attn_implementation
                    self.config._attn_implementation = "eager"
                    handle = decoder_layer.self_attn.register_forward_hook(capture_attn_hook, with_kwargs=True)

                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )

                if handle is not None:
                    handle.remove()
                    self.config._attn_implementation = prev_impl

                if should_prune and layer_idx == patcher.config.k:
                    attn_weights = captured_attn_weights.get("weights")
                    if attn_weights is not None:
                        vision_token_indices = image_mask[0].nonzero(as_tuple=True)[0]
                        keep_indices = patcher.ranker.select_tokens(
                            attn_weights, vision_token_indices, 1.0 - patcher.config.r
                        )
                        non_image_indices = (~image_mask[0]).nonzero(as_tuple=True)[0]
                        keep_indices = torch.cat([keep_indices, non_image_indices])
                        keep_indices, _ = torch.sort(keep_indices)

                        hidden_states = hidden_states[:, keep_indices]
                        position_ids = position_ids[:, keep_indices]
                        position_embeddings = tuple(pe[:, keep_indices] for pe in position_embeddings)
                        if attention_mask is not None:
                            attention_mask = attention_mask[:, keep_indices]
                        if past_key_values is not None:
                            for cache_layer in past_key_values.layers[: layer_idx + 1]:
                                cache_layer.keys = cache_layer.keys[:, :, keep_indices, :]
                                cache_layer.values = cache_layer.values[:, :, keep_indices, :]

                        causal_mask = create_causal_mask(
                            config=self.config,
                            inputs_embeds=hidden_states,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            position_ids=position_ids,
                            layer_idx=layer_idx + 1,
                        )
                    should_prune = False

            hidden_states = self.norm(hidden_states)

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
            )

        return patched_forward.__get__(language_model, type(language_model))
