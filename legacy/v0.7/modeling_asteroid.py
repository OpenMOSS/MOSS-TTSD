from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel, Qwen3Config, Qwen3Model
from transformers.cache_utils import Cache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import ModelOutput


class AsteroidTTSConfig(Qwen3Config):
    def __init__(
        self,
        channels=8,
        speech_pad_token=1024,
        speech_vocab_size=1025,
        speech_token_range=[],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.speech_pad_token = speech_pad_token
        self.speech_vocab_size = speech_vocab_size
        self.speech_token_range = speech_token_range


@dataclass
class AsteroidTTSOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    loss_all: Optional[Tuple[torch.FloatTensor]] = None
    logits_all: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


class CustomMixin(GenerationMixin):
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ) -> Union[GenerateDecoderOnlyOutput, torch.LongTensor]:
        # Extract configuration parameters
        speech_pad_idx = self.config.speech_pad_token

        eos_token_id = generation_config.eos_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        # Initialize output tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = (
            () if (return_dict_in_generate and output_attentions) else None
        )
        decoder_hidden_states = (
            () if (return_dict_in_generate and output_hidden_states) else None
        )

        # Initialize tracking variables
        batch_size, cur_len, channels = input_ids.shape  # channels = 8
        this_peer_finished = False
        unfinished_sequences = torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        needs_additional_steps = -1 * torch.ones(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        tf_inputs = input_ids[:]
        input_ids = input_ids[:, : -(channels - 1)]
        cur_len = input_ids.shape[1]
        model_kwargs["attention_mask"] = model_kwargs["attention_mask"][
            :, : -(channels - 1)
        ]
        base_length = input_ids.shape[1]
        model_kwargs = self._get_initial_cache_position(
            cur_len, input_ids.device, model_kwargs
        )

        # Define logits processor
        if generation_config.do_samples is not None:
            do_samples = generation_config.do_samples
            realprocessor = [LogitsProcessorList() for _ in range(channels)]
            for i, layer_config in enumerate(generation_config.layers):
                if layer_config.get("repetition_penalty") is not None:
                    realprocessor[i].append(
                        RepetitionPenaltyLogitsProcessor(
                            penalty=layer_config.get("repetition_penalty")
                        )
                    )
                if layer_config.get("temperature") is not None:
                    realprocessor[i].append(
                        TemperatureLogitsWarper(
                            temperature=layer_config.get("temperature")
                        )
                    )
                if layer_config.get("top_k") is not None:
                    realprocessor[i].append(
                        TopKLogitsWarper(top_k=layer_config.get("top_k"))
                    )
                if layer_config.get("top_p") is not None:
                    realprocessor[i].append(
                        TopPLogitsWarper(top_p=layer_config.get("top_p"))
                    )
        else:
            do_samples = [do_sample for _ in range(channels)]
            realprocessor = [logits_processor for _ in range(channels)]
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # Prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states}
                if output_hidden_states
                else {}
            )
            # Forward pass
            outputs = self(**model_inputs, return_dict=True)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs
            )

            if synced_gpus and this_peer_finished:
                continue

            # Get next token logits
            next_token_logits = [
                logits[:, -1, :].clone().float().to(input_ids.device)
                for logits in outputs.logits_all
            ]
            for i, channel_logits in enumerate(next_token_logits):
                if i != 0 and input_ids.shape[1] + 1 > tf_inputs.shape[1] - 7 + i:
                    channel_logits[:, 1024] = -torch.inf
                if i == 0 and input_ids.shape[1] + 1 <= tf_inputs.shape[1]:
                    channel_logits[:, 152694] = -torch.inf
            next_token_scores = [
                realprocessor[i](input_ids[..., i], logits)
                for i, logits in enumerate(next_token_logits)
            ]
            # Generate next tokens
            next_tokens = []
            for i, channel_score in enumerate(next_token_scores):
                if do_samples[i]:
                    channel_ntk = torch.multinomial(
                        nn.functional.softmax(channel_score, dim=-1), num_samples=1
                    ).squeeze(1)
                elif not do_samples[i]:
                    channel_ntk = torch.argmax(channel_score, dim=-1)
                next_tokens.append(channel_ntk)
            next_tokens = torch.stack(next_tokens, dim=-1)  # [batch_size, channels]
            # Additional steps logic
            indices = (~self.is_speech_token(next_tokens[:, 0])) & (
                needs_additional_steps < 0
            )
            needs_additional_steps[indices] = (
                channels - 1
            )  # For 8 channels, need 7 steps

            if input_ids.shape[1] + 1 <= tf_inputs.shape[1]:
                i = input_ids.shape[1] + 1 - base_length
                next_tokens[:, i:] = tf_inputs[:, input_ids.shape[1], i:]

            # Replace tokens in additional steps
            mask = (needs_additional_steps > 0) & (needs_additional_steps < 7)
            if mask.any().item():
                next_tokens[mask, 0] = self.config.eos_token_id
                for i in range(1, channels):
                    mask_i = mask & (needs_additional_steps < channels - i)
                    next_tokens[mask_i, i] = speech_pad_idx

            if has_eos_stopping_criteria:
                for i in range(channels):
                    pddp = self.config.eos_token_id if i == 0 else speech_pad_idx
                    next_tokens[:, i] = next_tokens[
                        :, i
                    ] * unfinished_sequences + pddp * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None, :]], dim=1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # Update unfinished_sequences
            needs_additional_steps = torch.where(
                needs_additional_steps > 0,
                needs_additional_steps - 1,
                needs_additional_steps,
            )
            stopping = stopping_criteria(input_ids[..., 0], scores) | (
                needs_additional_steps == 0
            )
            unfinished_sequences = unfinished_sequences & ~stopping
            unfinished_sequences = unfinished_sequences | (needs_additional_steps > 0)
            this_peer_finished = unfinished_sequences.max() == 0

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids


class AsteroidTTSPretrainedModel(PreTrainedModel):
    config_class = AsteroidTTSConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True


class AsteroidTTSModel(AsteroidTTSPretrainedModel):
    def __init__(self, config: AsteroidTTSConfig):
        super().__init__(config)
        self.text_pad_idx = config.pad_token_id
        self.speech_pad_idx = config.speech_pad_token
        self.embedding_list = nn.ModuleList([])
        self.embedding_list.append(
            nn.Embedding(config.vocab_size, config.hidden_size, self.text_pad_idx)
        )
        # Channels 1 to channels-1: Speech tokens only
        for _ in range(1, config.channels):
            self.embedding_list.append(
                nn.Embedding(
                    config.speech_vocab_size, config.hidden_size, self.speech_pad_idx
                )
            )

        self.language_model = Qwen3Model(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding_list[0]

    def set_input_embeddings(self, value: nn.Embedding):
        self.embedding_list[0] = value

    def _prepare_multi_modal_inputs(
        self, input_ids: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        Prepares multi-modal embeddings from input_ids of shape (batch_size, channels, sequence_length).
        For channel 0: text + speech tokens, for channels 1 to channels-1: speech tokens padded with speech_pad_token.
        """
        batch_size, seq_length, channels = input_ids.shape
        if channels != self.config.channels:
            raise ValueError(
                f"Expected {self.config.channels} channels, got {channels}"
            )

        inputs_embeds = torch.zeros(
            batch_size,
            seq_length,
            self.config.hidden_size,
            device=input_ids.device,
            dtype=self.embedding_list[0].weight.dtype,
        )
        for i in range(channels):
            embed_layer = self.embedding_list[i]
            channel_input = input_ids[..., i]
            inputs_embeds += embed_layer(channel_input)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,  # Shape: (batch_size, channels, sequence_length)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if input_ids is not None:
            inputs_embeds = self._prepare_multi_modal_inputs(input_ids)

        outputs = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        return outputs


class AsteroidTTSInstruct(AsteroidTTSPretrainedModel, CustomMixin):
    _tied_weights_keys = []
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: AsteroidTTSConfig):
        super().__init__(config)
        self.model = AsteroidTTSModel(config)
        self.channels = config.channels
        self.weights = [1 for _ in range(self.channels)]
        self._tied_weights_keys = [f"lm_heads.{i}.weight" for i in range(self.channels)]
        self.vocab_size = config.vocab_size
        self.lm_heads = nn.ModuleList([])
        self.lm_heads.append(
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )
        for _ in range(1, config.channels):
            self.lm_heads.append(
                nn.Linear(config.hidden_size, config.speech_vocab_size, bias=False)
            )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding_list[0]

    def can_generate(self):
        return True

    def is_speech_token(self, tokens):
        return (tokens >= self.config.speech_token_range[0]) & (
            tokens < self.config.speech_token_range[1]
        )

    def tie_weights(self):
        for i in range(self.config.channels):
            self._tie_or_clone_weights(self.lm_heads[i], self.model.embedding_list[i])

    def set_input_embeddings(self, value):
        self.model.embedding_list[0] = value

    def get_output_embeddings(self):
        return self.lm_heads[0]

    def set_output_embeddings(self, new_embeddings):
        self.lm_heads[0] = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def set_weights(self, weights):
        self.weights = weights

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        skip_logits: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, AsteroidTTSOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        skip_logits = (
            skip_logits
            if skip_logits is not None
            else (self.training and labels is not None)
        )
        if skip_logits and labels is None:
            skip_logits = False

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits_all = None
        loss_all = None
        total_loss = None

        if labels is not None:
            from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            loss_all = torch.empty(self.channels, device=device)
            logits_list = []

            for i in range(self.config.channels):
                vocab_size = (
                    self.config.vocab_size if i == 0 else self.config.speech_vocab_size
                )
                if skip_logits:
                    loss_all[i] = LigerForCausalLMLoss(
                        hidden_states=hidden_states,
                        lm_head_weight=self.lm_heads[i].weight,
                        labels=labels[..., i],
                        hidden_size=self.config.hidden_size,
                        **kwargs,
                    )
                else:
                    logits = self.lm_heads[i](hidden_states)
                    loss_all[i] = ForCausalLMLoss(logits, labels[..., i], vocab_size)
                    logits_list.append(logits)

            if not skip_logits:
                logits_all = tuple(logits_list)

            total_weight = sum(self.weights)
            normalized_weights = [w / total_weight for w in self.weights]

            total_loss = 0
            for w, loss in zip(normalized_weights, loss_all):
                total_loss += w * loss
        else:
            logits_all = [lm_head(hidden_states) for lm_head in self.lm_heads]

        if not return_dict:
            output = (logits_all,) + outputs[1:]
            return (
                (
                    total_loss,
                    loss_all,
                )
                + output
                if total_loss is not None
                else output
            )

        return AsteroidTTSOutputWithPast(
            loss=total_loss,
            logits=logits_all[0] if logits_all is not None else None,
            loss_all=loss_all,
            logits_all=logits_all,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
