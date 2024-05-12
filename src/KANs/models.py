import logging
from typing import (
    Optional,
    List,
    Union,
    Tuple
)
import torch
from torch import (
    nn, 
    FloatTensor, 
    Tensor, 
    LongTensor,
)
from transformers import (
    PretrainedConfig,
    ElectraPreTrainedModel,
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers.models.electra import (
    ElectraEmbeddings,
    ELECTRA_START_DOCSTRING,
    ELECTRA_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
)
from transformers.pytorch_utils import apply_chunking_to_forward
from kan import KANLayer


logger = logging.getLogger(__name__)


class KANElectraGeneator(ElectraPreTrainedModel):
    def __init__(
        self, 
        config: PretrainedConfig, 
    ) -> None:
        super().__init__()
        self.config = config
    
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str, 
        *model_args, 
        **kwargs
    ) -> "KANElectraGeneator":
        pass
    
    def forward(
        self, 
        input_ids: LongTensor, 
        attention_mask: LongTensor,
        token_type_ids: LongTensor,
        ) -> Tensor:
        pass
    
        

class KANElectraDiscriminator(ElectraPreTrainedModel):
    def __init__(
            self, 
            config: PretrainedConfig, 
        ) -> None:
        super().__init__()
    
    def forward(
        self, 
        input_ids: LongTensor, 
        attention_mask: LongTensor,
        token_type_ids: LongTensor,
    ):
        pass
    
    
@add_start_docstrings(
    "The bare Electra Model transformer outputting raw hidden-states without any specific head on top. Identical to "
    "the BERT model except that it uses an additional linear layer between the embedding layer and the encoder if the "
    "hidden size and embedding size are different. "
    ""
    "Both the generator and discriminator checkpoints may be loaded into this model.",
    "Fully connected layers are replaced into KAN Layers in this model.",
    ELECTRA_START_DOCSTRING,
)
class ElectraKANModel(ElectraPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.embeddings = ElectraEmbeddings(config)
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = KANLayer(config.embedding_size, config.hidden_size)
        self.encoder = ElectraKANEncoder(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.post_init()
    
    @property
    def input_embeddings(self):
        return self.embeddings.word_embeddings
    
    @input_embeddings.setter
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        output_type=BaseModelOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self, 
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
    ):
        batch_size, seq_length = input_ids.shape 
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_attention_mask=encoder_extended_attention_mask,
            return_dict=None,
        )
        return hidden_states
    

class ElectraKANEncoder(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.layer = nn.ModuleList([ElectraKANLayer(config) for _ in range(config.num_hidden_layers)])
        self.config = config
        self.gradient_checkpointing = False

        
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[FloatTensor] = None,
        head_mask: Optional[FloatTensor] = None,
        encoder_hidden_states: Optional[FloatTensor] = None,
        encoder_attention_mask: Optional[FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Tuple[torch.Tensor] | BaseModelOutputWithPastAndCrossAttentions:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class ElectraKANLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ElectraKANAttentionBlock(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ElectraKANAttentionBlock(config, position_embedding_type="absolute")
        self.intermediate = ElectraKANIntermediate(config)
        self.output = ElectraKANOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

