from typing import (
    Optional,
    List
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
)
from transformers.models.electra import (
    ElectraEmbeddings,
    ELECTRA_START_DOCSTRING,
    ELECTRA_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC
)
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
)

from kan import KANLayer


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
        self.config = config
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
        position_ids: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=None,
        )

        return hidden_states
    

class ElectraKANEncoder(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        