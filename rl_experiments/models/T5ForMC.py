import torch
import torch.nn as nn
from transformers import T5PreTrainedModel
from transformers.modeling_t5 import T5Stack
from torch.nn import CrossEntropyLoss
import copy

class T5ModelForMC(T5PreTrainedModel):
    """
    Wrapper for T5PreTrainedModel to use T5 for multiple choice under a closed choice set.
    - adds .QA_forward method
    
    (decoder) QA_forward
     Input:
        input_ids of shape: batch_size x num_choices x max_seq_len
     Output:
        outputs[0] is loss of shape batch_size x num_choices. preds should be torch.argmax(loss, dim = -1)

    """

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head


    def forward(self, **kwargs):
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.

        lm_labels = kwargs.pop("decoder_lm_labels", None)
        batch_loss = kwargs.pop("batch_loss", None)

        kwargs_common = dict(
            (k, v) for k, v in kwargs.items() if not k.startswith("encoder_") and not k.startswith("decoder_")
        )
        kwargs_encoder = kwargs_common.copy()
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder.update(dict((k[len("encoder_") :], v) for k, v in kwargs.items() if k.startswith("encoder_")))
        kwargs_decoder.update(dict((k[len("decoder_") :], v) for k, v in kwargs.items() if k.startswith("decoder_")))

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        if encoder_hidden_states is None:
            # Convert encoder inputs in embeddings if needed
            hidden_states = kwargs_encoder.pop("inputs_embeds", None)
            if hidden_states is None:
                encoder_inputs_ids = kwargs_encoder.pop("input_ids")
                hidden_states = self.shared(encoder_inputs_ids)  # Convert inputs in embeddings

            encoder_outputs = self.encoder(hidden_states, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[0]
        else:
            encoder_outputs = ()

        # Decode
        # Convert decoder inputs in embeddings if needed
        hidden_states = kwargs_decoder.pop("inputs_embeds", None)
        if hidden_states is None:
            decoder_inputs_ids = kwargs_decoder.pop("input_ids")
            hidden_states = self.shared(decoder_inputs_ids)

        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = kwargs_encoder.get("attention_mask", None)
        decoder_outputs = self.decoder(hidden_states, **kwargs_decoder)

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:            
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            if batch_loss:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
                real_label_lengths = (shift_labels != -100).sum(dim=-1, keepdim=True)
                loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
                loss = loss.sum(dim=-1, keepdim=True) / real_label_lengths
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # if loss != loss:
            #     print("got nan loss!")
            #
            #     loss_fct_unreduce = CrossEntropyLoss(ignore_index=-100, reduction = 'none')
            #     def nanmean(v, *args, inplace=False, **kwargs):
            #         if not inplace:
            #             v = v.clone()
            #         is_nan = torch.isnan(v)
            #         v[is_nan] = 0
            #         return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)
            #     losses = loss_fct_unreduce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            #     loss = nanmean(loss)

            decoder_outputs = (
                loss,
            ) + decoder_outputs  # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        return decoder_outputs + encoder_outputs


    def QA_forward(self, **kwargs):
        '''
        so this is basically just .forward that maintains the num_choices dimension, plus doesn't reduce the token loss into a scalar

        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.
        '''
        
        batch_size = kwargs['decoder_input_ids'].size(0)
        num_choices = kwargs['decoder_input_ids'].size(1)
        seq_len = kwargs['decoder_input_ids'].size(2)

        lm_labels = kwargs.pop("decoder_lm_labels", None)

        # kwargs_encoder/decoder are initialized from kwargs_common, and then overwritten by any encoder_/decoder_ prefixed arguments
        # arguments inside of kwargs_encoder/decoder are NOT prefixed
        kwargs_common = dict(
            (k, v) for k, v in kwargs.items() if not k.startswith("encoder_") and not k.startswith("decoder_")
        )
        kwargs_encoder = kwargs_common.copy()
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder.update(dict((k[len("encoder_") :], v) for k, v in kwargs.items() if k.startswith("encoder_")))
        kwargs_decoder.update(dict((k[len("decoder_") :], v) for k, v in kwargs.items() if k.startswith("decoder_")))

        # Encode if needed (training, first prediction pass)
        encoder_hidden_states = kwargs_encoder.pop("hidden_states", None)
        if encoder_hidden_states is None:
            # Convert encoder inputs in embeddings if needed
            hidden_states = kwargs_encoder.pop("inputs_embeds", None)
            if hidden_states is None:
                encoder_inputs_ids = kwargs_encoder.pop("input_ids")
                hidden_states = self.shared(encoder_inputs_ids)  # Convert inputs in embeddings

            encoder_outputs = self.encoder(hidden_states, **kwargs_encoder)
            encoder_hidden_states = encoder_outputs[0]
        else:
            encoder_outputs = ()

        # Decode
        # Convert decoder inputs in embeddings if needed
        hidden_states = kwargs_decoder.pop("inputs_embeds", None)
        if hidden_states is None:
            decoder_inputs_ids = kwargs_decoder.pop("input_ids")
            hidden_states = self.shared(decoder_inputs_ids)

        kwargs_decoder["encoder_hidden_states"] = encoder_hidden_states
        kwargs_decoder["encoder_attention_mask"] = kwargs_encoder.get("attention_mask", None)

        # have to combine batch_size and num_choices dimensions while preserving other dimensions for call to self.decoder
        hidden_states = hidden_states.view(-1, hidden_states.size(-2), hidden_states.size(-1)) if hidden_states is not None else None    
        for k, v in kwargs_decoder.items():
            if v.dim() == 3:
                kwargs_decoder[k] = v.reshape(-1, v.size(-1))
            elif v.dim() == 4:
                kwargs_decoder[k] = v.reshape(-1, v.size(-2), v.size(-1))

        decoder_outputs = self.decoder(hidden_states, **kwargs_decoder)

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output.reshape(batch_size, num_choices, seq_len, -1)
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        decoder_outputs = (lm_logits,) + decoder_outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:            

            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction = 'none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # reshape to batch_size x num choices x -1 then sum out the token dim (alternatively, could take the mean and not penalize longer answers)
            loss = loss.reshape(batch_size, num_choices, -1)
            loss = torch.mean(loss, dim=-1)

            decoder_outputs = (
                loss,
            ) + decoder_outputs  # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        return decoder_outputs + encoder_outputs


