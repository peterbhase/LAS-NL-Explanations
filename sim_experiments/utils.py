import os
from sacrebleu import corpus_bleu
import csv
import argparse
import logging
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

### METRICS ###

def computeBLEU(outputs, targets):
    # see https://github.com/mjpost/sacreBLEU
    targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return corpus_bleu(outputs, targets, lowercase=True).score

def CE_Loss(probabilities, labels):
    pred_probs = probabilities.gather(-1, labels.unsqueeze(-1))
    return torch.mean(-torch.log(pred_probs))

### END METRICS ###


### SAMPLING FUNCTIONS ###

def T5_sample(model, encoder_hidden_states, decoder_input_ids, encoder_attention_mask, tokenizer, max_sample_len):
        '''
        Uses model to sample based on context_ids, until max_sample_len is hit, with the expectation that decoding will stop at a specified [end] token
        This function is batched, meaning predictions are placed at the end of each running sequence within a tensor of shape (batch_size x num_choices x max_seq_len)        
        Before returning samples, the original contexts in running_contexts are set to the pad_token_id
        '''
        batch_size = decoder_input_ids.size(0)
        vocab_size = len(tokenizer) # NOT tokenizer.vocab_size, this attr does not update when tokens are added
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        eos_token_id = tokenizer.eos_token_id
        running_contexts = decoder_input_ids.clone()
        seq_len = decoder_input_ids.size(-1)
        device = decoder_input_ids.device
        start = time.time()
        context_len = (running_contexts!=pad_token_id).sum(dim=-1).max().item()
        if eos_token_id is not None: where_eos_sampled = [False] * batch_size

        # pad the input contexts up to max_sample_len
        if running_contexts.size(-1) < (max_sample_len+context_len):
            extend_by = (max_sample_len+context_len) - running_contexts.size(-1)
            extension_shape = [batch_size, extend_by]
            padding = pad_token_id * torch.ones(extension_shape, dtype = torch.long)
            padding = padding.to(running_contexts.device)
            running_contexts = torch.cat((running_contexts,padding),dim=-1)
            seq_len = max_sample_len+context_len

        where_last_tokens = (running_contexts!=pad_token_id).sum(-1) - 1
        
        # BEGIN SAMPLING 
        for k in range(max_sample_len):

            attention_mask = (running_contexts!=pad_token_id).float()

            # hold onto the starting point of sampling for each context
            if k==0: init_where_last_tokens = where_last_tokens

            with torch.no_grad():
                outputs = model(encoder_hidden_states = encoder_hidden_states, 
                                encoder_attention_mask = encoder_attention_mask,
                                decoder_input_ids = running_contexts,
                                decoder_attention_mask = attention_mask)
                logits = outputs[0]

                # get logits corresponding to last tokens in each sequence
                logits = torch.stack([logits[i, last_idx, :] for i, last_idx in enumerate(where_last_tokens)])
                preds = torch.argmax(logits, dim = -1)
                
            # assign preds to the first pad location in each running_contexts[i,j,:] sequence. check if eos_token sampled in each sequence
            for i in range(batch_size):
                last_token_index = where_last_tokens[i]
                running_contexts[i,last_token_index+1] = preds[i].item()
                if eos_token_id is not None:
                    if preds[i].item() == eos_token_id: where_eos_sampled[i] = True

            # if eos tokens sampled in every sequence, quit sampling
            if all(where_eos_sampled):
                break

            # iterate where_last_tokens
            where_last_tokens = where_last_tokens + 1
        
        # lastly, set the context portion of each sample to the pad_token_id
        samples = running_contexts
        for i in range(batch_size):
            end_of_context_index = init_where_last_tokens[i]
            samples[i,:(end_of_context_index+1)] = pad_token_id

        # print("sample time per input: %.2f" % ((time.time()-start)/batch_size))
        del outputs, logits

        return samples

def get_differentiable_explanations(speaker_model, listener_model, context_ids, tokenizer, max_sample_len, method = 'differentiable_argmax', eos_token_id = None,
                        input_ids = None, input_masks = None, encoder_hidden_states = None, listener_context_ids = None):
        '''
        - Differentiable decoding based on context_ids as the beginning of the output sequence. Context_ids of shape: batch_size x max_seq_len
        - Returns indices of the last 'valid' sample in each sequence (i.e. one right before first eos-token or pad-token)
        '''
        assert context_ids.dim() == 2, "Should be sampling one sequence per data point"
        # get accessible models in multi-gpu case
        if hasattr(speaker_model, 'module'):
            _speaker_model = speaker_model.module
        else:
            _speaker_model = speaker_model
        if hasattr(listener_model, 'module'):
            _listener_model = listener_model.module
        else:
            _listener_model = listener_model
        batch_size = context_ids.size(0)
        seq_len = context_ids.size(1)
        vocab_size = _speaker_model.lm_head.out_features # NOT tokenizer.vocab_size, that would lead to an error
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        running_contexts = context_ids.clone()
        device = context_ids.device
        start = time.time()
        softmax = torch.nn.Softmax(dim=-1)
        context_len = (running_contexts!=pad_token_id).sum(dim=-1).max().item()
        max_explanation_len = max_sample_len + context_len
        pad_token_embed = _speaker_model.shared(torch.tensor([pad_token_id]).to(context_ids.device))
        pad_token_embed = pad_token_embed.detach()
        if eos_token_id is not None: where_eos_sampled = [False] * batch_size

        # pad running_contexts up to max_sample_len
        if running_contexts.size(-1) < (max_explanation_len):
            expand_by = (max_explanation_len) - running_contexts.size(-1)
            padding = torch.tensor(pad_token_id).expand(batch_size,expand_by)
            padding = padding.to(running_contexts.device)
            return_running_contexts = torch.cat((running_contexts,padding),dim=-1)
            seq_len = return_running_contexts.size(-1)
        else:
            return_running_contexts = running_contexts.clone()

        if encoder_hidden_states is None:
            outputs = speaker_model(input_ids = input_ids, 
                                    attention_mask = input_masks)
            encoder_hidden_states = outputs[1]

        if listener_context_ids is not None:
            listener_embeds = _listener_model.shared(running_contexts.clone())
        else:
            # make context tensor
            listener_context = "My commonsense tells me that"
            listener_context_ids = torch.tensor(tokenizer.encode(listener_context), dtype = torch.long).to(running_contexts.device)
            listener_context_ids = listener_context_ids.unsqueeze(0).expand(batch_size,len(listener_context_ids))
            # pad context tensor up to max len
            max_explanation_len = listener_context_ids.size(-1) + max_sample_len
            expand_by = (max_explanation_len) - listener_context_ids.size(-1)
            padding = torch.tensor(pad_token_id).expand(batch_size,expand_by)
            padding = padding.to(running_contexts.device)
            listener_context_ids = torch.cat((listener_context_ids, padding),dim=-1)
            # look up embeddings
            listener_embeds = _listener_model.shared(listener_context_ids.clone())

            
        # BEGIN SAMPLING
        for k in range(max_sample_len): 

            # get locations of last non-pad tokens in each sequence for purposes of: getting predictions from logits, and updating running_contexts
            speaker_where_last_tokens = []
            listener_where_last_tokens = []
            for sequence in return_running_contexts.tolist():
                if pad_token_id in sequence:
                    speaker_where_last_tokens.append(sequence.index(pad_token_id)-1)
                else:
                    speaker_where_last_tokens.append(running_contexts.size(-1)-1)
            for sequence in listener_context_ids.tolist():
                if pad_token_id in sequence:
                    listener_where_last_tokens.append(sequence.index(pad_token_id)-1)
                else:
                    listener_where_last_tokens.append(listener_context_ids.size(-1)-1)   

            # make logits mask                         
            logits_mask = torch.zeros(batch_size, seq_len, vocab_size)     
            logits_mask = logits_mask.to(device)      
            for i in range(batch_size):
                last_token_index = speaker_where_last_tokens[i]
                logits_mask[i,last_token_index,:] = 1

            # hold onto the starting point of sampling for each contexts
            if k == 0: 
                return_sample_embeds = _speaker_model.shared(return_running_contexts.clone())
                running_decoder_input_embeds = return_sample_embeds.clone()
                embed_dim = running_decoder_input_embeds.size(-1)

            # forward pass
            outputs = speaker_model(encoder_hidden_states = encoder_hidden_states, 
                            encoder_attention_mask = input_masks,
                            decoder_inputs_embeds = running_decoder_input_embeds)
            logits = outputs[0]

            # get logits corresponding to last tokens in each sequence, then get preds
            logits = logits.view(batch_size, seq_len, vocab_size)             
            logits = logits * logits_mask
            logits = torch.sum(logits, dim = 1)
            preds = torch.argmax(logits, dim = -1)

            # get the predicted token's embeddings for both the speaker and the listener
            if method == 'differentiable_argmax':
                preds_onehot = differentiable_argmax(logits, temperature = 1)
                pred_speaker_embeds = torch.mm(preds_onehot, _speaker_model.shared.weight) # these get passed in as decoder_inputs_embeds at next step
                pred_listener_embeds = torch.mm(preds_onehot, _listener_model.shared.weight) # these will get returned to be passed to the listening simulator model

            if method == 'averaged_embeddings':
                # get hidden states for each last token
                probs = softmax(logits)

                # averaged predictions over model token input embeddings
                averaged_embeddings = torch.mm(probs, _speaker_model.shared.weight)                
                speaker_embeds = averaged_embeddings[preds, :]

            # assign preds to the first pad location in each running_contexts[i,j,:] sequence, and decoder_hidden_states to the running_decoder_input_embeds
            for i in range(batch_size):
                speaker_last_token_index = speaker_where_last_tokens[i]
                listener_last_token_index = listener_where_last_tokens[i]
                return_running_contexts[i,speaker_last_token_index+1] = preds[i].item()
                return_sample_embeds[i,speaker_last_token_index+1,:] = pred_speaker_embeds[i,:]
                listener_context_ids[i,listener_last_token_index+1] = preds[i].item()
                listener_embeds[i,listener_last_token_index+1,:] = pred_listener_embeds[i,:]
                if eos_token_id is not None:
                    if preds[i].item() == eos_token_id: 
                        where_eos_sampled[i] = True

            if eos_token_id is not None: 
                if all(where_eos_sampled): break

            # reassign decoder_input_embeds
            running_decoder_input_embeds = return_sample_embeds # .clone() appears inconsequential here

        # now we return a list of embeddings and token_ids.
        return_sample_embeds_list = []
        return_messages_list = []

        # for any samples in a sequence after the first eos-token or pad-token, record only up to the eos. keep track of explanation_lens
        context_ids_list = listener_context_ids.tolist()
        explanation_lens = []
        for i in range(batch_size):
            if eos_token_id in context_ids_list[i]: 
                begin_id = context_ids_list[i].index(eos_token_id)
            elif pad_token_id in context_ids_list[i]:
                begin_id = context_ids_list[i].index(pad_token_id)
            else:
                begin_id = None
            # keep up to eos
            if begin_id is not None:
                return_sample_embeds_list.append(listener_embeds[i,:begin_id,:]) 
                return_messages_list.append(listener_context_ids[i,:begin_id])
                explanation_lens.append(begin_id)
            # no eos, keep whole sequence 
            else:
                return_sample_embeds_list.append(listener_embeds[i,:,:])
                return_messages_list.append(listener_context_ids[i,:])
                explanation_lens.append(max_explanation_len)
        
        del outputs, logits, encoder_hidden_states, running_decoder_input_embeds
        if method == 'differentiable_argmax': del preds_onehot
        if method == 'averaged_embeddings':  del probs, averaged_embeddings, predicted_embeddings

        return return_sample_embeds_list, return_messages_list, explanation_lens


### END SAMPLING FUNCTIONS ###


### TOKENIZATION FUNCTIONS ###

def trim_unks(x):
    try: 
        unk_id = x.index('_end_</w>')
        return x[:unk_id]
    except:
        return x

def detok_batch(tokenizer, x, ignore_tokens = None, eos_token = None):
    '''
    - convert x into strings using tokenizer
    - x is either tensor of dim 2 or dim 3 or a .tolist() of such a tensor
    - stop decoding if eos_token hit, if eos_token provided
    - skip over tokens in ignore_tokens
    '''

    if ignore_tokens is not None:
        ignore_tokens_idx = tokenizer.convert_tokens_to_ids(ignore_tokens)
        ignore_tokens_idx += [-100,-1]
    else:
        ignore_tokens = []
        ignore_tokens_idx = [-100,-1]

    # if tokenizer.pad_token_id is None:
    ignore_tokens_idx += [0]
    if not isinstance(x, list):
        x = x.tolist()
    dim = 3 if isinstance(x[0][0], list) else 2
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
    texts = []

    for i in range(len(x)):    
        if dim == 2:
            current_idx = []
            for j in range(len(x[i])):
                current_id = x[i][j]
                if current_id == eos_token_id:
                    break
                elif current_id not in ignore_tokens_idx:
                    current_idx.append(current_id)
            decoded_sequence = tokenizer.decode(current_idx)
            # check if any ignore_tokens are in decoded_sequence. this is happening for some reason. many token_ids lead to [UNK], but [UNK] maps to id=100
            if any([ignore_token in decoded_sequence for ignore_token in ignore_tokens]):                    
                decoded_sequence = ' '.join([token for token in decoded_sequence.split() if token not in ignore_tokens])
            # APPEND
            texts.append(decoded_sequence)
        elif dim == 3:        
            decoded_sequences = []
            for j in range(len(x[i])):
                current_idx = []
                for k in range(len(x[i][j])):
                    current_id = x[i][j][k]
                    if current_id == eos_token_id:
                        break
                    elif current_id not in ignore_tokens_idx:
                        current_idx.append(current_id)

                decoded_sequence = tokenizer.decode(current_idx)

                # check if any ignore_tokens are in decoded_sequence. this is happening for some reason. many token_ids lead to [UNK], but [UNK] maps to id=100
                if any([ignore_token in decoded_sequence for ignore_token in ignore_tokens]):
                    decoded_sequence = ' '.join([token for token in decoded_sequence.split() if token not in ignore_tokens])
                # APPEND single decoding
                decoded_sequences.append(decoded_sequence)

            # APPEND list of n decodings
            texts.append(decoded_sequences)

    return texts


### END TOKENIZATION FUNCTIONS ###


### MISC ###

def str2bool(v):
    # used for boolean argparse values
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def isNaN(x):
    return (x!=x)

def gumbel_softmax(logits, temperature):
    """
    based on implementation here: https://github.com/dev4488/VAE_gumble_softmax/blob/master/vae_gumbel_softmax.py
    the point is that derivative of output is taken w.r.t. y_soft, which is a differentiable function of the logits
    """
    import ipdb; ipdb.set_trace()
    logits_shape = list(logits.shape)
    gumbel_softmax = F.gumbel_softmax(logits, tau=temperature, hard = False)
    y_soft = gumbel_softmax(logits, temperature)
    shape = y_soft.size()
    ind = y_soft.argmax(dim=-1)
    y_hard = torch.zeros_like(y_soft).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*y_soft.shape)
    y_hard = (y_hard - y_soft).detach() + y_soft
    y_hard = torch.argmax(y_hard, dim=-1)
    return y_hard

def differentiable_argmax(logits, temperature):
    """
    take argmax on forward pass; use softmax for backward pass
    """
    logits_shape = list(logits.shape)
    y_soft = F.softmax(logits / temperature, dim=-1)
    shape = y_soft.size()
    ind = y_soft.argmax(dim=-1)
    y_hard = torch.zeros_like(y_soft).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*y_soft.shape)
    y_hard = (y_hard - y_soft).detach() + y_soft
    return y_hard

def removeNonAscii(s): 
    if isinstance(s, str):
        return "".join(i for i in s if ord(i)<128)
    else:
        return s

def bootstrap_diff_in_means(means1, means2, boottimes=1e5):
    return

### END MISC ###