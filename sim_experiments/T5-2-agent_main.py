import os
import argparse
import random
import csv
import time
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from models.T5ForMC import T5ModelForMC
from transformers import T5Tokenizer, T5Config
from transformers import AdamW, get_linear_schedule_with_warmup

import utils, QA_data_utils, NLI_data_utils
from utils import str2bool

from classes import Report

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except:
    print("Not loading apex\n")


def load_data(args, data_name, tokenizer, all_sequential_samplers = False):
    '''
    returns pytorch dataloaders for train and eval data
    '''
    filter_explanations = None
    version = '1.0' if '1.0' in args.data_dir else '1.1'

    if data_name == 'QA':
        read_function = QA_data_utils.read_CQA
        prep_function = QA_data_utils.get_tensors_for_T5_split
        extension = 'csv'
    if data_name == 'NLI':
        read_function = NLI_data_utils.read_NLI
        prep_function = NLI_data_utils.get_tensors_for_T5_split
        extension = 'tsv'

    train_examples = read_function(args,
                            input_file = os.path.join(args.data_dir, 'train.%s' % extension), 
                            explanations_to_use = args.explanations_to_use, 
                            labels_to_use = args.labels_to_use,
                            version = version)
    dev_examples = read_function(args,
                            input_file = os.path.join(args.data_dir, 'dev.%s' % extension), 
                            explanations_to_use = args.explanations_to_use, 
                            labels_to_use = args.labels_to_use,
                            version = version)
    test_examples = read_function(args,
                            input_file = os.path.join(args.data_dir, 'test.%s' % extension), 
                            explanations_to_use = 'None' if data_name == 'QA' else args.explanations_to_use, 
                            labels_to_use = None if (data_name=='QA' and args.labels_to_use == 'label') else args.labels_to_use,
                            version = version)

    # eval on train data for debugging
    if args.eval_on_train:
        dev_examples = train_examples

    context = "My commonsense tells me that"
    context_len = len(tokenizer.encode(context))

    # convert examples to lists of tensors, and put into TensorDatasets then dataloaders. use_explanations is flag for excluded explanations in inputs
    train_tensors = prep_function(args, examples = train_examples, 
                                            tokenizer = tokenizer, 
                                            max_seq_length = args.max_seq_length, 
                                            condition_on_explanations = args.condition_on_explanations,
                                            multi_explanation = args.multi_explanation,
                                            spliced_explanation_len=context_len+args.max_sample_len)
    shuffle = True if not all_sequential_samplers else False
    train_dataloader = DataLoader(TensorDataset(*train_tensors), shuffle=shuffle, batch_size=args.train_batch_size, 
                num_workers = 4, pin_memory = True)
    
    dev_tensors = prep_function(args, examples = dev_examples, 
                                            tokenizer = tokenizer, 
                                            max_seq_length = args.max_seq_length, 
                                            condition_on_explanations = args.condition_on_explanations,
                                            multi_explanation = args.multi_explanation,
                                            spliced_explanation_len=context_len+args.max_sample_len)
    dev_dataloader = DataLoader(TensorDataset(*dev_tensors), shuffle=False, batch_size=args.train_batch_size, 
                num_workers = 4, pin_memory = True)
    
    test_tensors = prep_function(args, examples = test_examples, 
                                            tokenizer = tokenizer, 
                                            max_seq_length = args.max_seq_length, 
                                            condition_on_explanations = args.condition_on_explanations,
                                            multi_explanation = args.multi_explanation,
                                            spliced_explanation_len=context_len+args.max_sample_len)
    test_dataloader = DataLoader(TensorDataset(*test_tensors), shuffle=False, batch_size=args.train_batch_size, 
                num_workers = 4, pin_memory = True)
    
    return train_dataloader, dev_dataloader, test_dataloader


def load_model(args, device, tokenizer, multi_gpu = True, role = 'task', finetuned_path = None):
    if finetuned_path is None:
        if role == 'task':
            print(f"\nLoading non-finetuned model: {args.task_pretrained_name}...")    
        else:
            print(f"\nLoading non-finetuned model: {args.sim_pretrained_name}...")    
    elif finetuned_path is not None:
        print(f"\nLoading fine-tuned model: {finetuned_path}...")

    model_class = T5ModelForMC
    model = model_class.from_pretrained(args.task_pretrained_name if role == 'task' else args.sim_pretrained_name, 
        project_to_small = True,
        cache_dir = args.cache_dir)
    model.resize_token_embeddings(len(tokenizer))

    if finetuned_path is not None:  

        model_state_dict = torch.load(finetuned_path, map_location=lambda storage, loc: storage) # args for preventing memory leakage across gpus                
        model.load_state_dict(model_state_dict, strict = False)
        del model_state_dict

    model.to(device)
    return model


def prepare_optimizer(args, model, num_train_optimization_steps):
    '''returns optimizer'''

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, 
                      lr=args.lr, 
                      correct_bias=True)
    return optimizer


def prepare_optimizerS(args, models, num_train_optimization_steps):
    '''returns optimizerS'''
    return [prepare_optimizer(args, model, num_train_optimization_steps) for model in models]


def switch_model_grad_req(model, set_to = False):
    for m in model.modules():
        if hasattr(m, 'weight'):
            if hasattr(m.weight, 'requires_grad'):  
                m.weight.requires_grad = set_to
        if hasattr(m, 'bias'): 
            if hasattr(m.bias, 'requires_grad'):  
                m.bias.requires_grad = set_to


def train_or_eval_epoch(args, device, dataloader, stats_dict, multi_gpu, 
                task_model, sim_model, task_model_optimizer, sim_model_optimizer, schedulers, tokenizer, 
                split_name):
    '''runs one epoch. returns stats_dict. updates model parameters if training'''
    is_train = ('tr' in split_name)
    if is_train:
        # task_model.train() -- this done below
        sim_model.train()
    else:
        task_model.eval()
        sim_model.eval()

    # for accessing .shared, grab underlying model
    if hasattr(sim_model, 'module'):
        _sim_model = sim_model.module
    else:
        _sim_model = sim_model

    # ignore these in decoding
    ignore_tokens_list = [tokenizer.pad_token, '<start>', '[UNK]']

    # init stat vars
    task_loss_sum = 0
    sim_loss_sum = 0
    human_exp_loss_sum = 0
    task_acc_sum = 0
    sim_baseline_acc_sum = 0
    sim_XE_acc_sum, sim_X_acc_sum, sim_E_acc_sum = 0, 0, 0
    n_steps, n_data_points = 0, 0
    n_batches = len(dataloader)
    start_time = time.time()
    label_strs, agent1_strs = [], []
    task_preds_list = []
    sim_XE_preds_list = []
    sim_X_preds_list = []
    sim_E_preds_list = []
    task_loss, sim_loss, XE_loss, E_loss = [torch.tensor(0.).to(device) for i in range(4)] # placeholder init for these
    sampling_exps = args.pass_explanations or (args.human_exp_coef > 0 and split_name != 'tr')

    context = tokenizer.encode("My commonsense tells me that")
    explanation_len = len(context) + args.max_sample_len
    pad_token_id = tokenizer.pad_token_id

    for step, batch in enumerate(dataloader):
        print(f"  {split_name.capitalize()} | Step {step+1} / {n_batches}", end = '\r')

        # find last token locations on the fly
        task_input_ids = batch[0]
        where_padding_starts = []
        for sequence in task_input_ids.tolist():
            if pad_token_id in sequence:
                where_padding_starts.append(sequence.index(pad_token_id))
            else:
                where_padding_starts.append(args.max_seq_length - explanation_len)
        
        # unpack batch vars
        batch = [item.to(device) for item in batch]
        # unpack batch
        task_input_ids, task_input_masks, \
          task_answer_ids, task_answer_masks, task_answer_labels, \
          task_output_ids, task_output_masks, task_output_labels, task_choice_labels, \
          explanation_input_ids, explanation_input_masks, \
          explanation_output_ids, explanation_output_masks, explanation_output_labels, \
          explanation_context_ids, explanation_only_ids, explanation_lens = batch 

        # local vars
        batch_size = task_output_ids.size(0)
        num_choices = task_output_ids.size(1)  
        
        # first get NOISELESS predictions from the task model. i.e. turn off dropout. these needed as labels for simulator (plus gives task acc)
        with torch.no_grad():            
            task_model.eval()
            # encode
            outputs = task_model(input_ids = task_input_ids, 
                                 attention_mask = task_input_masks)
            encoder_hidden_states = outputs[1]                    
            # expand encoder_hidden_states along num_choices dim
            expand_shape = list(encoder_hidden_states.shape)
            expand_shape.insert(1, num_choices)    
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(expand_shape)
            _task_input_masks = task_input_masks.unsqueeze(1).expand_as(task_output_masks)
            outputs = task_model(encoder_hidden_states = encoder_hidden_states, 
                                    encoder_attention_mask = _task_input_masks,
                                    decoder_input_ids = task_output_ids, 
                                    decoder_lm_labels = task_output_labels)
            choice_losses = outputs[0]
            # get task preds
            labels = task_choice_labels.detach().cpu()
            choice_losses = choice_losses.detach().cpu()
            task_preds = torch.argmin(choice_losses, dim=-1)
            task_preds_list.extend(task_preds.tolist())
            n_correct = (task_preds==labels).sum().item()
            task_acc_sum += n_correct
            # set back to .train() if necessary
            if is_train: task_model.train()
        
        # forward passes
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:

            '''
            agent 1 optimization:
            - y|x
            - e|x (,y in rationalize)
            - neg. sim y|e
            - sim y|x,e (left until agent 2 opt.)
            '''

            switch_model_grad_req(_sim_model, set_to = False)

            # agent-1 task forward
            outputs = task_model(encoder_input_ids = task_input_ids, 
                                 encoder_attention_mask = task_input_masks,
                                 decoder_input_ids = task_answer_ids,
                                 decoder_lm_labels = task_answer_labels)
            task_loss = outputs[0] / args.grad_accumulation_factor 

            # agent-1 explain forward -- separate encode/decode because encoder_hidden_states used below
            outputs = task_model(input_ids = explanation_input_ids, 
                                attention_mask = explanation_input_masks)
            encoder_hidden_states = outputs[1]   
            outputs = task_model(encoder_hidden_states = encoder_hidden_states, 
                            encoder_attention_mask = explanation_input_masks,
                            decoder_input_ids = explanation_output_ids, 
                            decoder_lm_labels = explanation_output_labels, 
                            decoder_attention_mask = explanation_output_masks)
            explanation_loss = outputs[0] / args.grad_accumulation_factor

            # sample explanations
            if sampling_exps:
                sampling_grad_req = torch.enable_grad() if args.pass_explanations and is_train else torch.no_grad()
                with sampling_grad_req:
                    use_contexts = torch.stack(
                                [explanation_context_ids[i, task_preds[i], :] for i in range(batch_size)], dim = 0
                            )
                    outputs = utils.get_differentiable_explanations(
                            speaker_model = task_model, 
                            listener_model = sim_model,
                            context_ids = use_contexts, 
                            tokenizer = tokenizer, 
                            max_sample_len = args.max_sample_len, 
                            eos_token_id = tokenizer.eos_token_id,
                            encoder_hidden_states = encoder_hidden_states,    
                            input_masks = explanation_input_masks)   
                    agent1_explanation_embeds = outputs[0]
                    agent1_explanation_ids = outputs[1]       
                    explanation_lens = outputs[2]   

                # detokenize expl. labels and agent1 explanations
                trimmed_agent1_explanation_ids = [seq[len(context):] if len(seq) > len(context) else [pad_token_id] for seq in agent1_explanation_ids]
                batch_label_strs = utils.detok_batch(tokenizer, 
                                                explanation_only_ids, 
                                                ignore_tokens = ignore_tokens_list)
                batch_agent1_strs = utils.detok_batch(tokenizer, 
                                                trimmed_agent1_explanation_ids, 
                                                ignore_tokens = ignore_tokens_list,
                                                eos_token = tokenizer.eos_token)
                label_strs.extend(batch_label_strs)
                # batch_agent1_strs = [exp[len(context):] for exp in batch_agent1_strs] 
                agent1_strs.extend(batch_agent1_strs)

            # get targets for agent2
            if args.agent2_target == 'agent1':
                sim_decoder_inputs = torch.stack([task_output_ids[i,task_preds[i]] for i in range(batch_size)], dim =0)
                sim_decoder_labels = torch.stack([task_output_labels[i,task_preds[i]] for i in range(batch_size)], dim =0)
            elif args.agent2_target == 'task':
                sim_decoder_inputs = task_answer_ids.clone()
                sim_decoder_labels = task_answer_labels.clone()

            # get agent2 inputs (either pass explanations or use original -- only proceed with forward passes here if passing explanations)
            sim_encoder_inputs = _sim_model.shared(task_input_ids)
            sim_encoder_ids = task_input_ids.clone()
            sim_encoder_masks = task_input_masks.clone()        
            simulator_X_masks = sim_encoder_masks.clone()    
            if args.pass_explanations:
                for i in range(batch_size):
                    start_idx = where_padding_starts[i]
                    end_idx = start_idx + explanation_lens[i]    
                    sim_encoder_inputs[i,start_idx:end_idx,:] = agent1_explanation_embeds[i]
                    sim_encoder_ids[i,start_idx:end_idx] = agent1_explanation_ids[i]
                    sim_encoder_masks[i,start_idx:end_idx] = 1.

                # get encoder masks for agent2
                simulator_E_masks = sim_encoder_masks.clone()
                input_lens = (sim_encoder_ids!=pad_token_id).sum(-1).cpu()
                explanation_lens = torch.tensor(explanation_lens)
                x_lens = input_lens - explanation_lens - 4
                x_start_idx = len(tokenizer.encode('task: [CLS]') if 'v1.0' in args.data_dir else tokenizer.encode('nli premise: [CLS]'))
                for idx in range(batch_size):
                    simulator_E_masks[idx,x_start_idx:x_lens[idx]].fill_(0.)

                # sim XE pass for agent1
                outputs = sim_model(encoder_inputs_embeds = sim_encoder_inputs, 
                                encoder_attention_mask = sim_encoder_masks,
                                decoder_input_ids = sim_decoder_inputs, 
                                decoder_lm_labels = sim_decoder_labels)
                XE_loss = outputs[0] / args.grad_accumulation_factor 

                # sim E pass for agent1
                outputs = sim_model(encoder_inputs_embeds = sim_encoder_inputs, 
                                encoder_attention_mask = simulator_E_masks,
                                decoder_input_ids = sim_decoder_inputs, 
                                decoder_lm_labels = sim_decoder_labels)
                E_loss = outputs[0] / args.grad_accumulation_factor 


            # backprop and step if training
            if is_train:         

                # task loss can be up to three terms: (1) task loss, (2) explanation modeling loss, (3) sim loss
                task_loss = args.task_coef * task_loss + \
                            (args.human_exp_coef * explanation_loss if args.human_exp_coef > 0 else 0) + \
                            (1-args.task_coef-args.human_exp_coef) * ( (1-args.suppress_coef)*XE_loss - args.suppress_coef*E_loss )

                # backward
                if args.fp16:
                    for loss, optimizer in zip([task_loss, sim_loss], [task_model_optimizer, sim_model_optimizer]):
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                else:
                    task_loss.backward(retain_graph=True)

                # step
                if (step+1) % args.grad_accumulation_factor == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(task_model.parameters(), args.max_grad_norm)
                    task_model_optimizer.step()
                    task_model_optimizer.zero_grad()    

            # -- agent 2 opt. -- #

            '''
            agent2 gets three forward passes:
            - y|x,e
            - y|x
            - y|e
            '''
            
            switch_model_grad_req(_sim_model, set_to = True)
            sim_encoder_inputs = _sim_model.shared(sim_encoder_ids)

            if args.pass_explanations:
                
                # sim XE pass
                outputs = sim_model(encoder_inputs_embeds = sim_encoder_inputs, 
                                encoder_attention_mask = sim_encoder_masks,
                                decoder_input_ids = sim_decoder_inputs, 
                                decoder_lm_labels = sim_decoder_labels)
                XE_loss = outputs[0] / args.grad_accumulation_factor 
                encoder_hidden_states = outputs[2].detach()

                with torch.no_grad():
                    expand_shape = list(encoder_hidden_states.shape)
                    expand_shape.insert(1, num_choices)                
                    encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(expand_shape)
                    expand_shape = list(sim_encoder_masks.shape)
                    expand_shape.insert(1, num_choices)                
                    sim_encoder_masks = sim_encoder_masks.unsqueeze(1).expand(expand_shape)

                    outputs = sim_model(encoder_hidden_states = encoder_hidden_states, 
                                            encoder_attention_mask = sim_encoder_masks,
                                            decoder_input_ids = task_output_ids, 
                                            decoder_lm_labels = task_output_labels)
                    choice_losses = outputs[0]

                    # compute sim accuracy
                    choice_losses = choice_losses.detach().cpu()
                    sim_XE_preds = torch.argmin(choice_losses, dim=-1)
                    sim_XE_preds_list.extend(sim_XE_preds.tolist())
                    n_correct = (sim_XE_preds==task_preds).sum().item()
                    sim_XE_acc_sum += n_correct

                # sim E pass
                outputs = sim_model(encoder_inputs_embeds = sim_encoder_inputs, 
                                encoder_attention_mask = simulator_E_masks,
                                decoder_input_ids = sim_decoder_inputs, 
                                decoder_lm_labels = sim_decoder_labels)
                E_loss = outputs[0] / args.grad_accumulation_factor 
                encoder_hidden_states = outputs[2].detach()

                with torch.no_grad():
                    expand_shape = list(encoder_hidden_states.shape)
                    expand_shape.insert(1, num_choices)                
                    encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(expand_shape)
                    expand_shape = list(simulator_E_masks.shape)
                    expand_shape.insert(1, num_choices)                
                    simulator_E_masks = simulator_E_masks.unsqueeze(1).expand(expand_shape)

                    outputs = sim_model(encoder_hidden_states = encoder_hidden_states, 
                                        encoder_attention_mask = simulator_E_masks,
                                        decoder_input_ids = task_output_ids, 
                                        decoder_lm_labels = task_output_labels)
                    E_choice_losses = outputs[0]

                    # compute sim accuracy
                    E_choice_losses = E_choice_losses.detach().cpu()
                    sim_E_preds = torch.argmin(E_choice_losses, dim=-1)
                    sim_E_preds_list.extend(sim_E_preds.tolist())
                    n_correct = (sim_E_preds==task_preds).sum().item()
                    sim_E_acc_sum += n_correct


            # sim X pass
            outputs = sim_model(encoder_inputs_embeds = sim_encoder_inputs, 
                            encoder_attention_mask = simulator_X_masks,
                            decoder_input_ids = sim_decoder_inputs, 
                            decoder_lm_labels = sim_decoder_labels)
            X_loss = outputs[0] / args.grad_accumulation_factor 
            encoder_hidden_states = outputs[2].detach()

            with torch.no_grad():
                expand_shape = list(encoder_hidden_states.shape)
                expand_shape.insert(1, num_choices)                
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(expand_shape)
                expand_shape = list(simulator_X_masks.shape)
                expand_shape.insert(1, num_choices)                
                simulator_X_masks = simulator_X_masks.unsqueeze(1).expand(expand_shape)

                outputs = sim_model(encoder_hidden_states = encoder_hidden_states, 
                                        encoder_attention_mask = simulator_X_masks,
                                        decoder_input_ids = task_output_ids, 
                                        decoder_lm_labels = task_output_labels)
                X_choice_losses = outputs[0]

                # compute sim accuracy
                X_choice_losses = X_choice_losses.detach().cpu()
                sim_X_preds = torch.argmin(X_choice_losses, dim=-1)
                sim_X_preds_list.extend(sim_X_preds.tolist())
                n_correct = (sim_X_preds==task_preds).sum().item()
                sim_X_acc_sum += n_correct


            # gather losses if multi gpu
            if multi_gpu:
                task_loss = task_loss.mean()
                XE_loss = XE_loss.mean()
                X_loss = X_loss.mean()
                E_loss = E_loss.mean()
                # E_loss = E_loss.mean(dim=0) -- need to keep batch dim here
                if args.human_exp_coef > 0: explanation_loss = explanation_loss.mean()


            # backprop and step if training
            if is_train:         

                # sim loss is three terms: (1) yhat|x,e loss (2) yhat|x loss, (3) yhat|e loss
                sim_loss = (1-args.X_coef-args.E_coef)*XE_loss + args.X_coef*X_loss + args.E_coef*E_loss

                # backward
                if args.fp16:
                    for loss, optimizer in zip([task_loss, sim_loss], [task_model_optimizer, sim_model_optimizer]):
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(sim_model_optimizer), args.max_grad_norm)
                else:
                    sim_loss.backward()
                
                if (step+1) % args.grad_accumulation_factor == 0:    
                    torch.nn.utils.clip_grad_norm_(sim_model.parameters(), args.max_grad_norm)

                    sim_model_optimizer.step()
                    sim_model_optimizer.zero_grad()
                    for scheduler in schedulers: 
                        scheduler.step()

                    n_steps += 1   
                

            # track stats
            task_loss_sum += task_loss.item()
            sim_loss_sum += sim_loss.item()
            if args.human_exp_coef > 0: human_exp_loss_sum += explanation_loss.item()


            # clean up
            n_data_points += batch_size
            if is_train: 
                del task_loss, sim_loss, XE_loss, X_loss, E_loss
                del batch, outputs, encoder_hidden_states
                if args.human_exp_coef > 0: del explanation_loss

    # print examples
    if args.print_examples:
        print(f"\nEXAMPLE {split_name.upper()} INPUTS")
        num_to_print = min(batch_size, 3)
        input_strs = utils.detok_batch(tokenizer, task_input_ids, ignore_tokens = ignore_tokens_list)
        sim_input_strs = utils.detok_batch(tokenizer, sim_encoder_ids, ignore_tokens = ignore_tokens_list)
        answer_strs = utils.detok_batch(tokenizer, task_output_labels, ignore_tokens = ignore_tokens_list)
        explanation_id_strs = utils.detok_batch(tokenizer, explanation_output_ids, ignore_tokens = ignore_tokens_list)
        explanation_labels_strs = utils.detok_batch(tokenizer, explanation_output_labels, ignore_tokens = ignore_tokens_list)
        context_strs = utils.detok_batch(tokenizer, explanation_context_ids, ignore_tokens = ignore_tokens_list)
        for i in range(num_to_print):
            baseline_pred = sim_X_preds[i].int().item()
            print("task input:", input_strs[i])
            print(answer_strs[i][task_choice_labels[i].item()])
            print("model pred: ", answer_strs[i][task_preds[i]])
            print("simulator input:", sim_input_strs[i])
            print("baseline pred:", answer_strs[i][baseline_pred])
            if sampling_exps:
                print("human exp:", batch_label_strs[i])
                print("agent1 context:", context_strs[i][baseline_pred])
                print("agent1:", batch_agent1_strs[i])
            if args.pass_explanations:
                print("agent2 XE input: ", tokenizer.decode(sim_encoder_ids[i] * sim_encoder_masks[i][0]))
                print("agent2 E input: ", tokenizer.decode(sim_encoder_ids[i] * simulator_E_masks[i][0]))
            print("agent2 X input: ", tokenizer.decode(sim_encoder_ids[i] * simulator_X_masks[i][0]))
            print()
        # import ipdb; ipdb.set_trace()
        
    # summary stats    
    task_acc_mean = task_acc_sum / n_data_points
    sim_XE_acc_mean = sim_XE_acc_sum / n_data_points
    sim_X_acc_mean = sim_X_acc_sum / n_data_points
    sim_E_acc_mean = sim_E_acc_sum / n_data_points
    sim_baseline_acc_mean = sim_baseline_acc_sum / n_data_points
    task_loss_mean = task_loss_sum / n_batches
    sim_loss_mean = sim_loss_sum / n_batches
    human_exp_loss_mean = human_exp_loss_sum / n_batches

    # print('\n')
    # if split_name == 'tr':
    #     print(f"{split_name.upper()} LOSSES")
    #     print('task_loss : %.2f' % task_loss_mean)
    #     print('sim_loss : %.2f' % sim_loss_mean)
    #     print("task loss: %.4f" % task_loss)
    #     print("exp loss: %.4f" % explanation_loss)
    #     print("task loss: %.4f" % XE_loss)
    #     print("XE loss: %.4f" % XE_loss)
    #     print("X loss: %.4f" % X_loss)
        # print("E loss: %.4f" % E_loss.mean())
        # import ipdb; ipdb.set_trace()
        # print('\t human_exp_loss : %.2f' % human_exp_loss_mean)

    # do sim analysis, controlling for leaking
    labels = np.array(task_preds_list)
    xe = np.array(sim_XE_preds_list)
    x = np.array(sim_X_preds_list)
    e = np.array(sim_E_preds_list)
    xe_correct = np.array(1*(labels==xe))
    x_correct = np.array(1*(labels==x))
    e_correct = np.array(1*(labels==e))
    leaked = np.argwhere(e_correct)
    nonleaked = np.setdiff1d(np.arange(len(e_correct)), leaked)
    xe_correct_leaked = xe_correct[leaked]
    e_correct_leaked = e_correct[leaked]
    x_correct_leaked = x_correct[leaked]
    xe_correct_nonleaked = xe_correct[nonleaked]
    e_correct_nonleaked = e_correct[nonleaked]
    x_correct_nonleaked = x_correct[nonleaked]
    num_leaked = len(leaked)
    num_non_leaked = len(xe) - num_leaked
    baseline_correct = 1*(x_correct)
    unweighted_mean = np.mean([np.mean(xe_correct[split]) - np.mean(baseline_correct[split]) for split in [leaked,nonleaked]])
    if split_name in ['dv','test']:
        print("\n------------------------")
        print(f"{split_name.upper()} SIM ANALYSIS")
        print("num (probably) leaked: %d" % num_leaked)
        print("y|x,e : %.4f    baseline : %.4f     y|x,e=null: %.4f" % (np.mean(xe_correct_leaked), np.mean(baseline_correct[leaked]), np.mean(x_correct_leaked)))
        print("diff: %.4f" % (np.mean(xe_correct_leaked) - np.mean(baseline_correct[leaked])))
        print()
        print("num (probably) nonleaked: %d" % num_non_leaked)
        print("y|x,e : %.4f    baseline : %.4f     y|x,e=null: %.4f" % (np.mean(xe_correct_nonleaked), np.mean(baseline_correct[nonleaked]), np.mean(x_correct_nonleaked)))
        print("diff: %.4f" % (np.mean(xe_correct_nonleaked) - np.mean(baseline_correct[nonleaked])))
        print()
        print("overall: ")
        print("y|x : %.4f      y|e : %.4f" % (np.mean(x_correct), np.mean(e_correct)))
        print("y|x,e: %.4f     baseline : %.4f" % (np.mean(xe_correct), np.mean(baseline_correct)))
        print("\nunweighted mean: %.2f" % (unweighted_mean*100))

    stats = {}    
    bleu = -1
    stats.update({f'{split_name}_task_acc' : task_acc_mean * 100})
    stats.update({f'{split_name}_XE_acc' : sim_XE_acc_mean * 100})   
    stats.update({f'{split_name}_X_acc' : sim_X_acc_mean * 100})   
    stats.update({f'{split_name}_E_acc' : sim_E_acc_mean * 100})   
    stats.update({f'{split_name}_sim' : unweighted_mean * 100})   
    if sampling_exps:
        bleu = utils.computeBLEU(agent1_strs, [[x] for x in label_strs])
    stats.update({f'{split_name}_bleu' : bleu})
    stats_dict.update(stats)

    run_time = (time.time() - start_time) / 60
    print(f"\n  {split_name.capitalize()} time: {run_time:1.2f} minutes")
    
    return stats_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--task_pretrained_name", default='t5-base', type=str, help='HuggingFace transformer model')    
    parser.add_argument("--sim_pretrained_name", default='t5-small', type=str, help='HuggingFace transformer model')    
    parser.add_argument("--max_seq_length", default=110, type=int, help="The maximum total input sequence length after WordPiece tokenization. \n"
                                                                     "Sequences longer than this will be truncated, and sequences shorter \n"
                                                                     "than this will be padded.")
    # hyperparams
    parser.add_argument("--train_batch_size", default=2, type=int, help="Total batch size for training. Effective batch size is this times grad_accumulation_factor")
    parser.add_argument('--grad_accumulation_factor', type=int, default=6, help="Number of updates steps to accumulate before performing a backward pass and step.")
    parser.add_argument("--dev_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--lr", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")    
    parser.add_argument("--warmup_proportion", default=0.01, type=float, help="Proportion of training to perform linear learning rate warmup for. "
                                                                            "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument("--task_coef", default=1, type=float, help="Coefficient for task loss.")
    parser.add_argument("--human_exp_coef", default=0, type=float, help="Coefficient for explanation modeling loss.")
    parser.add_argument("--X_coef", default=0, type=float, help="Coefficient for y|x component of simulator loss.")
    parser.add_argument("--E_coef", default=0, type=float, help="Coefficient for y|e component of simulator loss.")
    parser.add_argument("--suppress_coef", default=.5, type=float, help="Coefficient for y|e component of AGENT1 loss, vs. y|x,e.")
    parser.add_argument('--max_sample_len', type = int, default = 20, help = 'Maximum num tokens that can appear in generated explanation')    
    # gpu + distributed + half-precision training
    parser.add_argument('--gpu', type = int, default = -1, help = 'gpu id to use. -1 defaults to multi-gpu')
    parser.add_argument('--fp16', default=False, type=str2bool, help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level',
                        type=str, default='O1',
                        help="Optimization level for mixed precision. Options are ['O0', 'O1', 'O2', and 'O3']. Details at See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    # misc
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--debug', action='store_true', help='Flag that queues ipdb before training')
    parser.add_argument('--print_examples', action='store_true', help='Flag that prints examples at end of each training/eval epoch')
    # directories + file paths
    parser.add_argument("--save_dir", default='/playpen3/home/peter/saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--data_dir', type=str, default='data/v1.0/', 
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--report_dir", default='training_reports/', type=str,
                    help="The output directory where the model training reports will be written.")
    parser.add_argument("--cache_dir", default='/playpen3/home/peter/cached_models/', type=str,
                    help="Directory for cacheing pretrained models.")
    parser.add_argument('--model_name', type=str, default = '',
                           help = "Save and/or load name for model. See below in script for exact formatting")
    parser.add_argument('--task_prefinetuned_name', type=str, default = '',
                           help = "Load name for model to start training with.")
    parser.add_argument('--sim_prefinetuned_name', type=str, default = '',
                           help = "Load name for model to start training with.")
    # debug flags
    parser.add_argument('--small_data', action='store_true', help='Flag for using just a few datapoints for debugging purposes')
    parser.add_argument('--small_size', default = 100, type=int, help='')
    # experiment condition flags
    parser.add_argument("--condition_on_explanations", default = False, type = str2bool,
                                help="Whether or not to condition on explanations in input")
    parser.add_argument("--explanations_to_use", default = 'oracle', choices = ['oracle', 't5','multi_t5', 'MT_t5', 'MT_multi_t5'], 
                                help="Which explanations to load with data.")
    parser.add_argument("--flip_exps", default = False, type=str2bool,  help="If true, write explanations then answers in input.")
    parser.add_argument("--labels_to_use", default = 'label',
                                help="Which labels to use with data. Intended for the use of simulating other models")
    parser.add_argument("--do_task", default = True, type=str2bool,  help="Do QA")
    parser.add_argument("--do_explain", default = True, type=str2bool,  help="Do LM")
    parser.add_argument("--select_for", default = 'sim', type=str,  help="Select model based on acc, bleu, or simulatability")
    parser.add_argument("--multi_explanation", default = True, type=str2bool,  help="Generate an explanation for each answer choice")
    parser.add_argument("--pass_explanations", default = True, type=str2bool,  help="If true, agent2 receives explanations from agent1 for task simulation")
    parser.add_argument("--agent2_target", default = 'agent1', type=str, choices =['agent1','task'],  help="primary target for agent 2")
    parser.add_argument("--opt_only_agent2", default=999, type=int, help = "Epoch to turn off optization for agent1 and optimize only agent 2")  
    # control flow for script
    parser.add_argument("--do_train", default = True, type=str2bool, help="Whether to run training.")
    parser.add_argument("--do_eval", default = True, type=str2bool, help="Whether to run final eval on dev and test sets.")    
    parser.add_argument("--eval_on_train",  default = False, action='store_true', help="Whether to run eval on the train data.")
    parser.add_argument("--pre_eval",  default = False, action='store_true', help="Evaluate once before training.")
    parser.add_argument('--eval_after', type = int, default = -1, help = 'checkpoint/evaluate models after this epoch')
    parser.add_argument('--load_epoch', type = str, default = 0, help = 'checkpoint/evaluate models after this epoch')
    parser.add_argument("--save_every_epoch", default = True, type=str2bool, help="Whether to run final eval on dev and test sets.")    
    
    # check argparse arguments. some argument settings don't make sense together
    args = parser.parse_args()    
    
    # GPU + SEED set-up
    n_gpu = torch.cuda.device_count()
    multi_gpu = (n_gpu > 1 and args.gpu == -1) # i.e. multiple gpus available and gpu choice not specified
    if multi_gpu: 
        device = torch.device("cuda") if args.gpu == -1 else torch.device(f'cuda:{args.gpu}')
        assert args.train_batch_size % n_gpu == 0, f"Train batch size will need to be allocated equally across {n_gpu} gpus, but {args.train_batch_size} cannot be"
        assert args.dev_batch_size % n_gpu == 0, f"Eval batch size will need to be allocated equally across {n_gpu} gpus, but {args.dev_batch_size} cannot be"
    else:
        device = torch.device(f"cuda:{args.gpu}")     
        torch.cuda.set_device(device)   
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if multi_gpu:
        torch.cuda.manual_seed_all(args.seed)

    # local variables
    if '1.0' in args.data_dir:
        data_name = 'QA'
    elif 'NLI' in args.data_dir:
        data_name = 'NLI'

    # make paths and dirs
    task_name = f"{data_name}_2-agent-task_{args.task_pretrained_name}_{args.model_name}_seed{args.seed}_epoch{args.load_epoch}"
    sim_name = f"{data_name}_2-agent-sim_{args.sim_pretrained_name}_{args.model_name}_seed{args.seed}_epoch{args.load_epoch}"
    task_model_path = os.path.join(args.save_dir, task_name + ".hdf5")    
    sim_model_path = os.path.join(args.save_dir, sim_name + ".hdf5")    
    task_prefinetuned_name = f"{data_name}_{args.task_pretrained_name}_{args.task_prefinetuned_name}"
    sim_prefinetuned_name = f"{data_name}_{args.sim_pretrained_name}_{args.sim_prefinetuned_name}"
    task_prefinetuned_path = os.path.join(args.save_dir, task_prefinetuned_name + ".hdf5") if args.task_prefinetuned_name != '' else None
    sim_prefinetuned_path = os.path.join(args.save_dir, sim_prefinetuned_name + ".hdf5") if args.sim_prefinetuned_name != '' else None
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    if not os.path.exists(args.report_dir): os.makedirs(args.report_dir)
    
    # make Report object + stats_dict
    report_name = f"report_{data_name}_2-agent_{args.model_name}_seed{args.seed}.txt"
    report_file = os.path.join(args.report_dir, report_name)
    score_names = ['tr_task_acc','tr_XE_acc','tr_X_acc','tr_E_acc','tr_sim', 'tr_bleu',
                   'dv_task_acc','dv_XE_acc','dv_X_acc','dv_E_acc','dv_sim', 'dv_bleu']
    report = Report(args, report_file, score_names = score_names)
    stats_dict = {}


    # LOAD TOKENIZER(s). note T5 tokenizer had pad and eos tokens by default
    tokenizer = T5Tokenizer.from_pretrained(args.task_pretrained_name, cache_dir = args.cache_dir)

    # LOAD DATA
    train_dataloader, dev_dataloader, test_dataloader = load_data(args, data_name, tokenizer)
    print(f"Data set sizes: \n Train: {len(train_dataloader.dataset)} \n Eval: {len(dev_dataloader.dataset)} \n Test: {len(test_dataloader.dataset)}")


    # flag so that model loaded before debug flag
    if args.do_train or args.pre_eval:

        # LOAD MODEL
        task_model = load_model(args, device, tokenizer, multi_gpu = multi_gpu, role = 'task', finetuned_path = task_prefinetuned_path)    
        sim_model = load_model(args, device, tokenizer, multi_gpu = multi_gpu, role = 'sim', finetuned_path = sim_prefinetuned_path)    
        
        # LOAD OPTIMIZERs
        num_train_optimization_steps = args.num_train_epochs * int(len(train_dataloader.dataset) / args.train_batch_size / args.grad_accumulation_factor)
        optimizers = prepare_optimizerS(args, models = [task_model,sim_model], num_train_optimization_steps = num_train_optimization_steps)
        task_model_optimizer, sim_model_optimizer = optimizers
        
        # mixed precision version of models + optimizers
        if args.fp16:
            task_model, sim_model, optimizer = amp.initialize([task_model, sim_model], optimizer, opt_level=args.fp16_opt_level)
            
        # MAKE SCHEDULER -- needs to occur after amp.initialize due to https://discuss.pytorch.org/t/cyclic-learning-rate-how-to-use/53796
        schedulers = [get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps= int(args.warmup_proportion * num_train_optimization_steps), 
                                                num_training_steps=num_train_optimization_steps)       
                      for optimizer in optimizers]

        # models to multi_gpu (needs to follow mixed-precision)
        if multi_gpu:
            task_model = torch.nn.DataParallel(task_model, device_ids = range(n_gpu))
            sim_model = torch.nn.DataParallel(sim_model, device_ids = range(n_gpu))

    # debug flag
    if args.debug:
        import ipdb; ipdb.set_trace()

    if args.pre_eval:
        stats_dict = train_or_eval_epoch(args, device, dev_dataloader, stats_dict, multi_gpu = multi_gpu, 
                       task_model = task_model, 
                       sim_model = sim_model,
                       task_model_optimizer = task_model_optimizer, 
                       sim_model_optimizer = sim_model_optimizer,
                       schedulers = None, 
                       tokenizer = tokenizer, 
                       split_name = 'dv')   
        report.print_epoch_scores(epoch = -1, scores = stats_dict)
        stats_dict = {}

    # BEGIN TRAINING
    best_epoch = -1.0
    best_score = -1.0
    if args.do_train:

        # training loop
        print("\nBeginning training...\n")
        start_time = time.time()
        for e in range(args.num_train_epochs):
            print(f"Epoch {e}")
            print("LR: %.6f" % task_model_optimizer.param_groups[0]['lr'])

            stats_dict = train_or_eval_epoch(args, device, train_dataloader, stats_dict, multi_gpu = multi_gpu, 
                                    task_model = task_model, 
                                    sim_model = sim_model,
                                    task_model_optimizer = task_model_optimizer, 
                                    sim_model_optimizer = sim_model_optimizer,
                                    schedulers = schedulers, 
                                    tokenizer = tokenizer,
                                    split_name = 'tr')   
            if e > args.eval_after:
                stats_dict = train_or_eval_epoch(args, device, dev_dataloader, stats_dict, multi_gpu = multi_gpu, 
                                       task_model = task_model, 
                                       sim_model = sim_model,
                                       task_model_optimizer = task_model_optimizer, 
                                       sim_model_optimizer = sim_model_optimizer,
                                       schedulers = None, 
                                       tokenizer = tokenizer,
                                       split_name = 'dv')   
                score = stats_dict['dv_' + args.select_for]        
    
                # check for best dev score and save if new best
                if score > best_score or args.save_every_epoch:
                    print(f"  \tSaving model(s) in {args.save_dir}")
                    task_name = f"{data_name}_2-agent-task_{args.task_pretrained_name}_{args.model_name}_seed{args.seed}_epoch{e}"
                    sim_name = f"{data_name}_2-agent-sim_{args.sim_pretrained_name}_{args.model_name}_seed{args.seed}_epoch{e}"
                    task_model_path = os.path.join(args.save_dir, task_name + ".hdf5")    
                    sim_model_path = os.path.join(args.save_dir, sim_name + ".hdf5")    
                    model_to_save = task_model.module if hasattr(task_model, 'module') else task_model 
                    torch.save(model_to_save.state_dict(), task_model_path)
                    model_to_save = sim_model.module if hasattr(sim_model, 'module') else sim_model 
                    torch.save(model_to_save.state_dict(), sim_model_path)
                    if score > best_score:
                        best_score = score
                        best_epoch = e 
                
            # write + print summary stats
            report.write_epoch_scores(epoch = e, scores = stats_dict)
            report.print_epoch_scores(epoch = e, scores = stats_dict)

        end_time = time.time()
        training_time = (end_time-start_time) / 60
        unit = 'minutes' if training_time < 60 else 'hours'
        training_time = training_time if training_time < 60 else training_time / 60
        print(f"\nTotal training time: {training_time:.2f} {unit}")

    # FINAL EVAL
    task_model = load_model(args, device, tokenizer, multi_gpu = multi_gpu, role = 'task', finetuned_path = task_model_path)
    sim_model = load_model(args, device, tokenizer, multi_gpu = multi_gpu, role = 'sim', finetuned_path = sim_model_path)
    if multi_gpu:
        task_model = torch.nn.DataParallel(task_model)
        sim_model = torch.nn.DataParallel(sim_model)
    
    if args.do_eval:
        print("\nGetting final eval results...\n")
        stats_dict = train_or_eval_epoch(args, device, dev_dataloader, stats_dict, multi_gpu = multi_gpu, 
                               task_model = task_model, 
                               sim_model = sim_model,
                               task_model_optimizer = None, 
                               sim_model_optimizer = None,
                               schedulers = None, 
                               tokenizer = tokenizer, 
                               split_name = 'dv')   
        stats_dict = train_or_eval_epoch(args, device, test_dataloader, stats_dict, multi_gpu = multi_gpu, 
                           task_model = task_model, 
                           sim_model = sim_model,
                           task_model_optimizer = None, 
                           sim_model_optimizer = None,
                           schedulers = None, 
                           tokenizer = tokenizer,
                           split_name = 'test')       
        
        # print and write final stats
        dev_sim_acc, test_sim_acc, dev_task_acc, test_task_acc, dev_bleu, test_bleu, dev_agent_bleu, test_agent_bleu = -1, -1, -1, -1, -1, -1, -1, -1
        dev_sim_acc = stats_dict['dv_XE_acc']        
        test_sim_acc = stats_dict['test_XE_acc']
        dev_task_acc = stats_dict['dv_task_acc']        
        test_task_acc = stats_dict['test_task_acc']
        dev_bleu = stats_dict['dv_bleu']
        test_bleu = stats_dict['test_bleu']        
        final_msg = f"Best epoch: {best_epoch}" + \
                    f" | Dev task acc: {dev_task_acc:.2f} | Dev sim-XE acc: {dev_sim_acc:.2f} | Dev BLEU: {dev_bleu:.2f} " + \
                    f"\n Test task acc: {test_task_acc:.2f} | Test sim acc: {test_sim_acc:.2f} | Test BLEU: {test_bleu:.2f}"
        if args.do_train:
            report.write_final_score(args, final_score_str = final_msg)
        report.print_epoch_scores(epoch = best_epoch, scores = {k:v for k,v in stats_dict.items() if 'train' not in k})



    ### END OF SCRIPT ###


