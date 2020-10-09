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

    def __init__(self, config, project_to_small = False):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if project_to_small:
            self.project_to_small = nn.Linear(768, 512) # projection matrix for use in self.project_base_to_small

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head


    def forward(self, reduce_batch = True, loss_weights = None, **kwargs):
        # keyword arguments come in 3 flavors: encoder-specific (prefixed by
        # `encoder_`), decoder-specific (prefixed by `decoder_`) and those
        # that apply to the model as whole.
        # We let the specific kwargs override the common ones in case of conflict.

        # IF decoder_input_ids has NUM CHOICES dimension, return .QA_forward. this exists so we can wrap model with torch.nn.DataParallel, which must call forward
        if 'decoder_input_ids' in kwargs.keys():
            if kwargs['decoder_input_ids'].dim() == 3: # batch_size x num_choices x max_seq_len
                return self.QA_forward(**kwargs)


        lm_labels = kwargs.pop("decoder_lm_labels", None)

        kwargs_common = dict(
            (k, v) for k, v in kwargs.items() if not k.startswith("encoder_") and not k.startswith("decoder_")
        )
        kwargs_encoder = kwargs_common.copy()
        kwargs_decoder = kwargs_common.copy()
        kwargs_encoder.update(dict((k[len("encoder_") :], v) for k, v in kwargs.items() if k.startswith("encoder_")))
        kwargs_decoder.update(dict((k[len("decoder_") :], v) for k, v in kwargs.items() if k.startswith("decoder_")))

        # import ipdb; ipdb.set_trace()

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
        # print(kwargs_decoder.keys())
        # print(kwargs_decoder["encoder_hidden_states"].shape)
        # print(kwargs_decoder["encoder_attention_mask"].shape)  
        # print(hidden_states.shape)
        # import ipdb; ipdb.set_trace()
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
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction = 'none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # reshape to batch shape (shifted)
            batch_shape = shift_labels.shape
            loss = loss.view(batch_shape)
            # get per data point loss
            loss = torch.mean(loss, dim=-1)                
            if reduce_batch:
                loss = loss.mean()
            if loss_weights is not None:
                loss = loss * loss_weights

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

        # combine batch_size and num_choices dimension in all inputs
        for kwargs in [kwargs_encoder, kwargs_decoder]:
            for k, v in kwargs.items():
                if hasattr(v,'dim'):
                    if v.dim() == 3:
                        kwargs[k] = v.reshape(-1, v.size(-1))
                    elif v.dim() == 4:
                        kwargs[k] = v.reshape(-1, v.size(-2), v.size(-1))        

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
            # reshape to batch_size x num choices x -1 then take mean over the token dim
            loss = loss.reshape(batch_size, num_choices, -1)
            loss = torch.mean(loss, dim=-1)

            decoder_outputs = (
                loss,
            ) + decoder_outputs  # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        return decoder_outputs + encoder_outputs


    def project_base_to_small(self, embeddings):
        '''
        written for 2-agent experiments with task model as t5-base and simulator as t5-small.
        this will project a set of embeddings down to t5-small's embedding dim
        '''
        return self.project_to_small(embeddings)