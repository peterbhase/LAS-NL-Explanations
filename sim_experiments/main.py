import os
import argparse
import random
import csv
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from models.T5ForMC import T5ModelForMC
from transformers import T5Tokenizer, T5Config, AutoTokenizer, AutoConfig
from transformers import RobertaForSequenceClassification, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup

import utils, QA_data_utils, NLI_data_utils
from utils import str2bool

from classes import Report

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except:
    print("Not loading apex\n")

def CE_Loss(probabilities, labels):
    pred_probs = probabilities.gather(-1, labels.unsqueeze(-1))
    return torch.mean(-torch.log(pred_probs))

def load_data(args, data_name, tokenizer):
    '''
    returns pytorch dataloaders for train and eval data
    '''
    filter_explanations = None
    version = '1.0' #if '1.0' in args.data_dir else '1.1'

    if data_name == 'QA':
        read_function = QA_data_utils.read_CQA
        if 't5' in args.task_pretrained_name:
            prep_function = QA_data_utils.get_tensors_for_T5_split
        elif 'bert' in args.task_pretrained_name:
            prep_function = QA_data_utils.get_tensors_for_bert
        extension = 'csv'
    if data_name == 'NLI':        
        read_function = NLI_data_utils.read_NLI
        if 't5' in args.task_pretrained_name:
            prep_function = NLI_data_utils.get_tensors_for_T5_split
        elif 'bert' in args.task_pretrained_name:
            prep_function = NLI_data_utils.get_tensors_for_bert
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
                            explanations_to_use = args.explanations_to_use, 
                            labels_to_use = None if (data_name=='QA' and args.labels_to_use == 'label') else args.labels_to_use,
                            version = version)

    # eval on train data for debugging
    if args.eval_on_train:
        dev_examples = train_examples

    # convert examples to lists of tensors, and put into TensorDatasets then dataloaders. use_explanations is flag for excluded explanations in inputs
    train_tensors = prep_function(args, examples = train_examples, 
                                            tokenizer = tokenizer, 
                                            max_seq_length = args.max_seq_length, 
                                            condition_on_explanations = args.condition_on_explanations,
                                            multi_explanation = args.multi_explanation,
                                            explanations_only = args.explanations_only)
    train_dataloader = DataLoader(TensorDataset(*train_tensors), shuffle=True, batch_size=args.train_batch_size if args.do_train else args.dev_batch_size, 
                num_workers = 4, pin_memory = True)
    sequential_train_dataloader = DataLoader(TensorDataset(*train_tensors), shuffle=False, batch_size=args.dev_batch_size, 
                num_workers = 4, pin_memory = True)
    
    dev_tensors = prep_function(args, examples = dev_examples, 
                                            tokenizer = tokenizer, 
                                            max_seq_length = args.max_seq_length, 
                                            condition_on_explanations = args.condition_on_explanations,
                                            multi_explanation = args.multi_explanation,
                                            explanations_only = args.explanations_only)
    dev_dataloader = DataLoader(TensorDataset(*dev_tensors), shuffle=False, batch_size=args.train_batch_size if args.do_train else args.dev_batch_size, 
                num_workers = 4, pin_memory = True)
    
    test_tensors = prep_function(args, examples = test_examples, 
                                            tokenizer = tokenizer, 
                                            max_seq_length = args.max_seq_length, 
                                            condition_on_explanations = args.condition_on_explanations,
                                            multi_explanation = args.multi_explanation,
                                            explanations_only = args.explanations_only)
    test_dataloader = DataLoader(TensorDataset(*test_tensors), shuffle=False, batch_size=args.train_batch_size if args.do_train else args.dev_batch_size, 
                num_workers = 4, pin_memory = True)
    
    return train_dataloader, dev_dataloader, test_dataloader, sequential_train_dataloader


def load_model(args, device, tokenizer, multi_gpu = True, finetuned_path = None):
    if finetuned_path is None:
        print(f"\nLoading non-finetuned model: {args.task_pretrained_name}...")
    elif finetuned_path is not None:
        print(f"\nLoading fine-tuned model: {finetuned_path}...")

    if 'bert' in args.task_pretrained_name:
        config = RobertaConfig.from_pretrained(args.task_pretrained_name, num_labels=3)
        model = RobertaForSequenceClassification.from_pretrained(args.task_pretrained_name, config=config, cache_dir = args.cache_dir)

    if 't5' in args.task_pretrained_name:
        model_class = T5ModelForMC
        model = model_class.from_pretrained(args.task_pretrained_name, 
            project_to_small=False,
            cache_dir = args.cache_dir)
        model.resize_token_embeddings(len(tokenizer))
        
    if finetuned_path is not None:
        model_state_dict = torch.load(finetuned_path, map_location=lambda storage, loc: storage) # args for preventing memory leakage across gpus
        model.load_state_dict(model_state_dict)    
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


def train_or_eval_epoch(args, device, dataloader, stats_dict, multi_gpu, 
                model, optimizer, scheduler, tokenizer, 
                sample_exps, split_name, write_predictions = False):
    '''runs one epoch. returns stats_dict. updates model parameters if training'''
    is_train = (split_name == 'train' and not write_predictions)
    if is_train:
        model.train()
    else:
        model.eval()

    # used when conditioning on explanations
    allow_dropout = (is_train or args.dropout_on_dev)
    # ST-RA
    ST_RA = (args.condition_on_explanations and args.multi_explanation)

    # ignore these in decoding
    ignore_tokens_list = [tokenizer.pad_token, '[UNK]']
    pad_token_id = tokenizer.pad_token_id

    # init stat vars
    task_loss_sum = 0
    explanation_loss_sum = 0
    acc_sum = 0
    n_steps, n_data_points = 0, 0
    n_batches = len(dataloader)
    start_time = time.time()
    task_loss, explanation_loss = torch.tensor([0.]), torch.tensor([0.]) # placeholders updated depending on experiment flow
    label_strs, sample_strs, multi_sample_strs = [], [], []
    preds_list = []
    label_probs_list = []

    for step, batch in enumerate(dataloader):
        print(f"  {split_name.capitalize()} | Step {step+1} / {n_batches}", end = '\r')

        # unpack batch vars
        batch = [item.to(device) for item in batch]
        task_input_ids, task_input_masks, \
          task_answer_ids, task_answer_masks, task_answer_labels, \
          task_output_ids, task_output_masks, task_output_labels, task_choice_labels, \
          explanation_input_ids, explanation_input_masks, \
          explanation_output_ids, explanation_output_masks, explanation_output_labels, \
          explanation_context_ids, explanation_only_ids, explanation_lens = batch

        # shape vars
        batch_size = task_output_ids.size(0)
        num_choices = 3

        # randomly dropping out explanations
        if args.explanation_dropout > 0 and allow_dropout:
            num_to_dropout = int(args.explanation_dropout * batch_size)
            eligible_idx = np.arange(batch_size)
            # eligible_idx = np.setdiff1d(np.arange(batch_size), np.argwhere(0==leaking.cpu().numpy())) # don't drop-out non-leaking cases
            # num_to_dropout = int(args.explanation_dropout*leaking.sum().item())
            # explanation_lens = (explanation_context_ids[:,0,:]!=pad_token_id).sum(-1) + (explanation_only_ids!=pad_token_id).sum(-1)
            input_lens = (task_input_ids!=pad_token_id).sum(-1)
            explanation_start_idx = input_lens-explanation_lens+1
            if 'bert' in args.task_pretrained_name:
                explanation_start_idx -= 1
            dropout_idx = np.random.choice(eligible_idx, replace=False, size = num_to_dropout)
            for idx in dropout_idx:
                task_input_ids[idx,explanation_start_idx[idx]:].fill_(pad_token_id)
            # print('\nexplanation dropout!')
            # print("dropout", dropout_idx)
            # # print("non-leaking", np.argwhere(0==leaking.cpu().numpy()))
            # for i, task_input in enumerate(task_input_ids.tolist()):
            #     print(i)
            #     print(tokenizer.decode(task_input, skip_special_tokens=True))
            # import ipdb; ipdb.set_trace()
        if args.input_dropout > 0 and allow_dropout:
            num_to_dropout = int(args.input_dropout * batch_size)            
            # if combining with explanation_dropout, never dropout both x and e for a data point
            if args.explanation_dropout > 0:
                eligible_idx = np.setdiff1d(np.arange(batch_size),dropout_idx)
            else:
                eligible_idx = np.arange(batch_size)
            # explanation_lens = (explanation_context_ids[:,0,:]!=pad_token_id).sum(-1) + (explanation_only_ids!=pad_token_id).sum(-1)
            input_lens = (task_input_ids!=pad_token_id).sum(-1)
            x_lens = input_lens - explanation_lens - 3
            if 't5' in args.task_pretrained_name:
                x_start_idx = len(tokenizer.encode('task: [CLS]') if 'v1.0' in args.data_dir else tokenizer.encode('nli premise: [CLS]'))
            elif 'bert' in args.task_pretrained_name:
                x_start_idx = 1
                x_lens = x_lens + 2
            dropout_idx = np.random.choice(eligible_idx, replace=False, size = num_to_dropout)
            for idx in dropout_idx:
                task_input_ids[idx,x_start_idx:x_lens[idx]].fill_(pad_token_id)
            # print("\ninput dropout!")
            # print("dropout", dropout_idx)
            # for i, task_input in enumerate(task_input_ids.tolist()):
            #     print(i)
            #     print(tokenizer.decode(task_input, skip_special_tokens=True))
            # import ipdb; ipdb.set_trace()

        # FORWARD
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:

            if args.do_task:                
                if 't5' in args.task_pretrained_name and not ST_RA:
                    outputs = model(input_ids = task_input_ids, 
                                attention_mask = task_input_masks)
                    encoder_hidden_states = outputs[1]  
                    outputs = model(encoder_hidden_states = encoder_hidden_states, 
                                    encoder_attention_mask = task_input_masks,
                                    decoder_input_ids = task_answer_ids, 
                                    decoder_lm_labels = task_answer_labels, 
                                    decoder_attention_mask = task_answer_masks)
                    task_loss = outputs[0] / args.grad_accumulation_factor 
                    choice_losses = None                
                    # now get likelihoods for each choice 
                    with torch.no_grad():
                        # add num_choices dim to input_masks and encoder_hidden_states and expand to match task_output_ids shape
                        expand_shape = list(encoder_hidden_states.shape)
                        expand_shape.insert(1, num_choices)    
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(expand_shape)
                        task_input_masks = task_input_masks.unsqueeze(1).expand_as(task_output_masks)

                        outputs = model(encoder_hidden_states = encoder_hidden_states, 
                                                encoder_attention_mask = task_input_masks,
                                                decoder_input_ids = task_output_ids, 
                                                decoder_lm_labels = task_output_labels, 
                                                decoder_attention_mask = task_output_masks)
                        # choice_losses is of shape: batch_size x num_choices, because task_output_ids had a num_choices dim
                        choice_losses = outputs[0]
                elif 't5' in args.task_pretrained_name and ST_RA:
                    batch_shape = list(task_input_ids.shape)
                    task_input_ids = task_input_ids.view(-1,task_input_ids.size(-1))
                    task_input_masks = task_input_masks.view(-1,task_input_ids.size(-1))
                    outputs = model(input_ids = task_input_ids, 
                                            attention_mask = task_input_masks)
                    encoder_hidden_states = outputs[1]       
                    # reshape inputs
                    task_input_ids = task_input_ids.view(batch_shape)
                    task_input_masks = task_input_masks.view(batch_shape)                
                    batch_shape.append(encoder_hidden_states.size(-1))
                    encoder_hidden_states = encoder_hidden_states.view(batch_shape)
                    outputs = model(encoder_hidden_states = encoder_hidden_states, 
                                encoder_attention_mask = task_input_masks,
                                decoder_input_ids = task_output_ids, 
                                decoder_lm_labels = task_output_labels, 
                                decoder_attention_mask = task_output_masks)
                    choice_losses = outputs[0] # choice_losses is of shape: batch_size x num_choices
                    choice_probs = nn.functional.softmax(-choice_losses, dim=-1)
                    task_loss = utils.CE_Loss(choice_probs, task_choice_labels) / args.grad_accumulation_factor
                elif 'bert' in args.task_pretrained_name:
                    outputs = model(input_ids = task_input_ids, 
                            attention_mask = task_input_masks,
                            labels = task_choice_labels)
                    choice_losses = -outputs[1] # negative logits, preds gotten by argmin
                    task_loss = outputs[0] / args.grad_accumulation_factor 
                    # print(task_loss)
             
                # compute task accuracy
                labels = task_choice_labels.detach().cpu().numpy()
                # choice_losses = choice_losses.detach().cpu().numpy()
                preds = np.argmin(choice_losses.detach().cpu().numpy(), axis=-1)
                n_correct = np.sum(preds==labels)
                acc_sum += n_correct 
                preds_list.extend(preds.tolist())

                # get pred probs
                choice_probs = nn.functional.softmax(-choice_losses.detach(), dim=-1)
                label_probs = [choice_probs[i,label].item() for i, label in enumerate(labels)]
                label_probs_list.extend(label_probs)

            if args.do_explain:
                outputs = model(input_ids = explanation_input_ids,
                                encoder_attention_mask = explanation_input_masks,
                                decoder_input_ids = explanation_output_ids, 
                                decoder_lm_labels = explanation_output_labels, 
                                decoder_attention_mask = explanation_output_masks)
                explanation_loss = outputs[0] / args.grad_accumulation_factor
                encoder_hidden_states = outputs[2]

            if multi_gpu:
                task_loss = task_loss.mean()
                explanation_loss = explanation_loss.mean()

            # BACKWARD
            if is_train:         
                loss = (args.task_coef * task_loss if args.do_task else 0) + \
                      ((1 - args.task_coef) * explanation_loss if args.do_explain else 0)
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()                        
                # step
                if (step+1) % args.grad_accumulation_factor == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    n_steps += 1
                    # print("stepping!")     

            # explanation sampling. sample when do_explain is true and either writing predictions or evaluating
            if sample_exps:
                if args.do_task: # get predicted contexts
                    use_contexts = torch.stack(
                            [explanation_context_ids[i, preds[i], :] for i in range(batch_size)], dim = 0
                        ).unsqueeze(1)
                elif args.multi_explanation and not write_predictions: # get correct contexts
                    use_contexts = torch.stack(
                            [explanation_context_ids[i, task_choice_labels[i], :] for i in range(batch_size)], dim = 0
                        ).unsqueeze(1)
                elif args.multi_explanation and write_predictions: # use all three contexts
                    use_contexts = explanation_context_ids
                elif not args.multi_explanation: # take an arbitrary context for each data point (all the same)
                    use_contexts = explanation_context_ids[:,0,:]

                # sample
                reshape = False
                if use_contexts.dim() == 3:               
                    first_two_dims = list(use_contexts.shape)[:2]
                    explanation_input_masks = explanation_input_masks.unsqueeze(1).expand_as(use_contexts)
                    expand_shape = list(encoder_hidden_states.shape)
                    expand_shape.insert(1, use_contexts.size(1))
                    encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(expand_shape)
                    use_contexts = use_contexts.view(-1,use_contexts.size(-1))
                    encoder_hidden_states = encoder_hidden_states.reshape(-1, encoder_hidden_states.size(-2), encoder_hidden_states.size(-1))
                    explanation_input_masks = explanation_input_masks.reshape(-1, explanation_input_masks.size(-1))
                    reshape = True
                samples = utils.T5_sample(model, 
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_input_ids=use_contexts,
                    encoder_attention_mask=explanation_input_masks, 
                    tokenizer=tokenizer, 
                    max_sample_len=args.max_sample_len)
                if reshape:
                    samples = samples.view(first_two_dims + [samples.size(-1)])

                if not args.do_task and args.multi_explanation and write_predictions: # condition where three are sampled per item
                    pred_explanations = [question[task_choice_labels[i].item()] for i, question in enumerate(samples.tolist())]
                    batch_multi_sample_strs = utils.detok_batch(tokenizer, samples, 
                                    ignore_tokens = ignore_tokens_list,
                                    eos_token = tokenizer.eos_token)
                    multi_sample_strs.extend(batch_multi_sample_strs)
                else:
                    pred_explanations = samples.squeeze(1).tolist()

                # detokenize expl. labels and predictions
                batch_label_strs = utils.detok_batch(tokenizer, explanation_only_ids, 
                                                ignore_tokens = ignore_tokens_list)
                batch_sample_strs = utils.detok_batch(tokenizer, pred_explanations, 
                                                ignore_tokens = ignore_tokens_list,
                                                eos_token = tokenizer.eos_token)
                label_strs.extend(batch_label_strs)
                sample_strs.extend(batch_sample_strs)

            # track stats
            task_loss_sum += task_loss.item()
            explanation_loss_sum += explanation_loss.item()
            n_data_points += batch_size

            # clean up
            if is_train:
                del loss
                if args.do_task: del task_loss
                if args.do_explain: del explanation_loss
                del batch, outputs
                
    # print examples
    if args.print_examples:
        print(f"\nEXAMPLE {split_name.upper()} INPUTS")
        num_to_print = min(batch_size, 6)
        input_strs = utils.detok_batch(tokenizer, task_input_ids, ignore_tokens = ignore_tokens_list)
        if 't5' in args.task_pretrained_name:
            answer_strs = utils.detok_batch(tokenizer, task_output_labels, ignore_tokens = ignore_tokens_list)
        else:
            answer_strs = [[str(num) for num in range(3)] for label in task_choice_labels]
        if sample_exps:
            used_contexts = utils.detok_batch(tokenizer, use_contexts, ignore_tokens = ignore_tokens_list)
        for i in range(num_to_print):
            print("Model input:", input_strs[i])
            print("True answer: ", answer_strs[i][task_choice_labels[i].item()])
            if args.do_task: print("Model answer: ", answer_strs[i][preds[i]])
            if sample_exps:
                print("human:", batch_label_strs[i])
                if not args.do_task and args.multi_explanation and write_predictions:
                    for j in range(num_choices):
                        print(f"context {j}: ", tokenizer.decode(explanation_context_ids[i][j]))
                        print(f"sample {j}:", batch_multi_sample_strs[i][j])
                    import ipdb; ipdb.set_trace() 
                print("context (bleu):", used_contexts[i])                
                print("sample (bleu):", batch_sample_strs[i])
            print()

    # summary stats
    task_loss_mean = task_loss_sum / n_data_points
    acc_mean = acc_sum / n_data_points
    explanation_loss_mean = explanation_loss_sum / n_data_points

    stats = {}
    if args.do_task:
        if split_name != 'test': stats.update({f'{split_name}_task_loss' : task_loss_mean})
        stats.update({f'{split_name}_acc' : acc_mean * 100})
    if args.do_explain:        
        if split_name != 'test': 
            explanation_loss_mean = explanation_loss_sum / n_batches        
            stats.update({f'{split_name}_exp_loss' : explanation_loss_mean})
        if sample_exps:
            bleu = utils.computeBLEU(sample_strs, [[x] for x in label_strs]) if len(sample_strs) > 0 else -1
            stats.update({f'{split_name}_bleu' : bleu})
    stats_dict.update(stats)

    run_time = (time.time() - start_time) / 60
    print(f"\n  {split_name.capitalize()} time: {run_time:1.2f} minutes")

    if write_predictions:
        extension = 'tsv' if 'NLI' in args.data_dir else 'csv'
        delimiter = '\t' if 'NLI' in args.data_dir else ','
 
        if args.do_task:
            df_path = os.path.join(args.data_dir, f'{split_name}.{extension}')
            df = pd.read_csv(df_path, sep=delimiter)
            n = len(df)
            new_col_name = f'preds_{save_name}' if args.preds_suffix is None else f'preds_{save_name}_{args.preds_suffix}'
            while len(preds_list) < n:
                preds_list.append('N/A')
            df[new_col_name] = preds_list
            if 'sim' in args.model_name.lower():
                new_col_name = f'label_probs_{save_name}' if args.preds_suffix is None else f'label_probs_{save_name}_{args.preds_suffix}'
                while len(label_probs_list) < n:
                    label_probs_list.append('N/A')
                df[new_col_name] = label_probs_list
            df.to_csv(df_path, index = False, sep = delimiter)

        if args.do_explain:
            df_path = os.path.join(args.data_dir, f'{split_name}.{extension}')
            df = pd.read_csv(df_path,sep=delimiter)
            n = len(df)

            if args.multi_explanation and args.do_task:
                col_name = f't5-MT-multi-exp-pred-seed{args.seed}' if not args.save_agent else 't5-agent-ra-exp'
                while len(sample_strs) < n:
                    sample_strs.append('N/A')
                df[col_name] = sample_strs

            if args.multi_explanation and not args.do_task:
                explanations = np.array(multi_sample_strs)
                exp_cols = [f't5-multi-exp-{i}-seed{args.seed}' for i in range(num_choices)]
                for j, col_name in enumerate(exp_cols):
                    new_col = explanations[:,j].tolist()
                    while len(new_col) < n:
                        new_col.append('N/A')
                    df[col_name] = new_col
            
            if not args.multi_explanation:
                if args.do_task:
                    col_name = f't5-MT-single-exp-seed{args.seed}'  if not args.save_agent else 't5-agent-re-exp'
                else:
                    col_name = f't5-single-exp-seed{args.seed}' 

                while len(sample_strs) < n:
                    sample_strs.append('N/A')

                df[col_name] = sample_strs

            df.to_csv(df_path, index = False, sep=delimiter)
    
    return stats_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--task_pretrained_name", default='t5-base', type=str, help='HuggingFace transformer model')    
    parser.add_argument("--max_seq_length", default=175, type=int, help="The maximum total input sequence length after WordPiece tokenization. \n"
                                                                     "Sequences longer than this will be truncated, and sequences shorter \n"
                                                                     "than this will be padded.")
    # hyperparams
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training. Effective batch size is this times grad_accumulation_factor")
    parser.add_argument('--grad_accumulation_factor', type=int, default=3, help="Number of updates steps to accumulate before performing a backward pass and step.")
    parser.add_argument("--dev_batch_size", default=20, type=int, help="Total batch size for eval.")
    parser.add_argument("--lr", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")    
    parser.add_argument("--warmup_proportion", default=0.01, type=float, help="Proportion of training to perform linear learning rate warmup for. "
                                                                            "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument("--task_coef", default=1, type=float, help="Coefficient for task loss.")
    parser.add_argument('--max_sample_len', type = int, default = 175, help = 'Maximum num tokens that can appear in generated explanation')    
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
    parser.add_argument("--save_dir", default='', required=True, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--data_dir', type=str, default='data/e-SNLI-data/',
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--report_dir", default='training_reports/', type=str,
                    help="The output directory where the model training reports will be written.")
    parser.add_argument("--cache_dir", default='', required=True, type=str,
                    help="Directory for cacheing pretrained models.")
    parser.add_argument('--model_name', type=str, default = 'unnamed',
                           help = "Save and/or load name for model. See below in script for exact formatting")
    parser.add_argument('--prefinetuned_name', type=str, default = '',
                           help = "Load name for model to start training with.")
    # debug flags
    parser.add_argument('--small_data', '-s', action='store_true', help='Flag for using just a few datapoints for debugging purposes')
    parser.add_argument("--small_size", '-ss', default=100, type=int, help = "")  
    # experiment condition flags
    parser.add_argument("--condition_on_explanations", default = False, type = str2bool,
                                help="Whether or not to condition on explanations in input")
    parser.add_argument("--explanations_to_use", default = 'ground_truth', help="Which explanations to load with data.")
    parser.add_argument("--explanations_only", default = False, type=str2bool,  help="Include only answer choices and explanations (no x) as input")
    parser.add_argument("--preds_suffix", default = None, type=str, choices=['X', 'XE', 'E'],  help="Indicator for input contents for a simulator model")
    parser.add_argument("--labels_to_use", default = 'label',
                                help="Which labels to use with data. Intended for the use of simulating other models")
    parser.add_argument("--do_task", default = True, type=str2bool,  help="Do QA")
    parser.add_argument("--do_explain", default = True, type=str2bool,  help="Do LM")
    parser.add_argument("--select_for", default = 'acc', type=str, choices=['acc', 'bleu'],  help="Select model based on acc or bleu")
    parser.add_argument("--multi_explanation", default = True, type=str2bool,  help="Generate an explanation for each answer choice")
    parser.add_argument("--leaking_weight", default = -1, type=int,  
                        help="Used if > 0 and conditioning on exps. Weight loss by whether exps leak labels. More heavily weight non-leaking examples")
    parser.add_argument("--leakage_predictor", default = None, type=str, help="Model y|e whose correctness indicates explanations leak label")
    parser.add_argument("--explanation_dropout", default = 0, type=float, help="When condition_on_explanations, proportion of exps to dropout from inputs")
    parser.add_argument("--input_dropout", default = 0, type=float, help="When condition_on_explanations, proportion of x to dropout from inputs")
    parser.add_argument("--dropout_on_dev", default = False, type=str2bool, help="Whether to run input/exp dropout on dev, if running dropout.")
    # control flow for script
    parser.add_argument("--do_train", default = True, type=str2bool, help="Whether to run training.")
    parser.add_argument("--save_agent", default = False, type=str2bool, help="Whether to run training.")
    parser.add_argument("--do_eval", default = True, type=str2bool, help="Whether to run final eval on dev and test sets.")    
    parser.add_argument("--eval_on_train",  default = False, action='store_true', help="Whether to run eval on the train data.")
    parser.add_argument('--write_predictions', action='store_true', default = False, help = 'Write predictions in data file')
    parser.add_argument("--load_epoch", default=0, type=int, help = "Epoch to effectively start at.")  
    parser.add_argument('--pre_eval', action='store_true', default = False, help = 'Evaluate model once before training')
    
    # check argparse arguments. some argument settings don't make sense together
    args = parser.parse_args()    
    assert args.do_task + args.do_explain >= 1, "Don't do nothing"
    assert not (args.do_explain and args.task_coef == 1) or not args.do_train, \
        "If explaining, use args.task_coef < 1 which implies explain_coef > 0"
    
    # GPU + SEED set-up
    n_gpu = torch.cuda.device_count()
    multi_gpu = (n_gpu > 1 and args.gpu == -1) # i.e. multiple gpus available and gpu choice not specified
    if multi_gpu: 
        device = torch.device("cuda") if args.gpu == -1 else torch.device(f'cuda:{args.gpu}')
        assert args.train_batch_size % n_gpu == 0, f"Train batch size will need to be allocated equally across {n_gpu} gpus, but {args.train_batch_size} cannot be"
        assert args.test_batch_size % n_gpu == 0, f"Eval batch size will need to be allocated equally across {n_gpu} gpus, but {args.dev_batch_size} cannot be"
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)   
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if multi_gpu:
        torch.cuda.manual_seed_all(args.seed)

    # local variables
    if '1.0' in args.data_dir or 'qa' in args.model_name:
        data_name = 'QA'
    elif 'NLI' in args.data_dir or 'nli' in args.model_name:
        data_name = 'NLI'
        args.max_seq_length = 128 # override to have lower max_seq_len
        args.max_sample_len = 128 # doesn't need to be higher than max_seq_length, naturally
        print("Overriding sequence length to %d and sample_len to %d" % (args.max_seq_length, args.max_sample_len))

    # make paths and dirs
    if data_name == 'QA':
        agent_insert = '2-agent-task_' if args.save_agent else ''
        agent_epoch = f'_epoch{args.load_epoch}' if args.save_agent else ''
        save_name = f"{data_name}_{agent_insert}{args.task_pretrained_name}_{args.model_name}_seed{args.seed}{agent_epoch}"

    elif data_name == 'NLI':
        agent_insert = '2-agent-task_' if args.save_agent else ''
        agent_epoch = f'_epoch{args.load_epoch}' if args.save_agent else ''
        save_name = f"{data_name}_{agent_insert}{args.task_pretrained_name}_{args.model_name}_seed{args.seed}{agent_epoch}"

    if args.small_data:
        save_name += '_DEBUG'

    print("Starting experiment with save_name: %s" % save_name)

    model_path = os.path.join(args.save_dir, save_name + ".hdf5")    
    prefinetuned_name = f"{data_name}_{args.task_pretrained_name}_{args.prefinetuned_name}"
    prefinetuned_path = os.path.join(args.save_dir, prefinetuned_name + ".hdf5") if args.prefinetuned_name != '' else None
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    if not os.path.exists(args.report_dir): os.makedirs(args.report_dir)

    # make Report object + stats_dict
    report_name = f"report_{save_name}.txt"
    report_file = os.path.join(args.report_dir, report_name)
    if args.do_task and not args.do_explain:
        score_names = ['train_task_loss','train_acc','dev_task_loss','dev_acc','test_task_loss','test_acc'] 
    elif args.do_explain and not args.do_task:
        score_names = ['train_exp_loss','dev_exp_loss','dev_bleu','test_bleu'] 
    else:
        score_names = ['train_task_loss','train_acc','train_exp_loss',
                       'dev_task_loss','dev_acc','dev_exp_loss','dev_bleu',
                       'test_acc','test_bleu'] 
    report = Report(args, report_file, score_names = score_names)
    stats_dict = {}

    # LOAD TOKENIZER(s). note T5 tokenizer had pad and eos tokens by default
    tokenizer = AutoTokenizer.from_pretrained(args.task_pretrained_name, cache_dir = args.cache_dir)        

    # LOAD DATA
    print("Loading data...")
    train_dataloader, dev_dataloader, test_dataloader, sequential_train_dataloader = load_data(args, data_name, tokenizer)
    print(f"Data set sizes: \n Train: {len(train_dataloader.dataset)} \n Eval: {len(dev_dataloader.dataset)} \n Test: {len(test_dataloader.dataset)}")

    # flag so that model loaded before debug flag
    if args.do_train:

        # LOAD MODEL
        model = load_model(args, device, tokenizer, multi_gpu = multi_gpu, finetuned_path = prefinetuned_path)    
        
        # LOAD OPTIMIZER
        num_train_optimization_steps = args.num_train_epochs * int(len(train_dataloader.dataset) / args.train_batch_size / args.grad_accumulation_factor)
        optimizer = prepare_optimizer(args, model = model, num_train_optimization_steps = num_train_optimization_steps)
        
        # mixed precision version of models + optimizers
        if args.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
            
        # MAKE SCHEDULERS -- needs to occur after amp.initialize due to https://discuss.pytorch.org/t/cyclic-learning-rate-how-to-use/53796
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps= int(args.warmup_proportion * num_train_optimization_steps), 
                                                num_training_steps=num_train_optimization_steps)        

        # models to multi_gpu (needs to follow mixed-precision)
        if multi_gpu:
            model = torch.nn.DataParallel(model, device_ids = range(n_gpu))

    # debug flag
    if args.debug:
        import ipdb; ipdb.set_trace()

    if args.pre_eval:
        stats_dict = train_or_eval_epoch(args, device, dev_dataloader, stats_dict, multi_gpu = multi_gpu, 
                       model = model,
                       optimizer = None, 
                       scheduler = None, 
                       tokenizer = tokenizer,
                       sample_exps = False, 
                       split_name = 'dev')   
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
            print("LR: %.6f" % optimizer.param_groups[0]['lr'])

            stats_dict = train_or_eval_epoch(args, device, train_dataloader, stats_dict, multi_gpu = multi_gpu, 
                                    model = model, 
                                    optimizer = optimizer, 
                                    scheduler = scheduler, 
                                    tokenizer = tokenizer,
                                    sample_exps = False,
                                    split_name = 'train')   
            stats_dict = train_or_eval_epoch(args, device, dev_dataloader, stats_dict, multi_gpu = multi_gpu, 
                                   model = model, 
                                   optimizer = None, 
                                   scheduler = None, 
                                   tokenizer = tokenizer,
                                   sample_exps = (not args.do_task and args.do_explain),
                                   split_name = 'dev')   
            score = stats_dict['dev_' + args.select_for]        
                
            # check for best dev score and save if new best
            if score > best_score:
                print(f"  New best model. Saving model in {args.save_dir}")
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
                torch.save(model_to_save.state_dict(), model_path)
                best_score = score
                best_epoch = e                                 
            
            # write + print summary stats
            report.write_epoch_scores(epoch = e, scores = stats_dict)
            report.print_epoch_scores(epoch = e, scores = stats_dict)

        end_time = time.time()
        training_time = (end_time-start_time) / 60
        unit = 'minutes' if training_time < 60 else 'hours'
        training_time = training_time if training_time < 60 else training_time / 60
        time_msg = f"\nTotal training time: {training_time:.2f} {unit}"
        print(time_msg)

    # FINAL EVAL

    model = load_model(args, device, tokenizer, multi_gpu = multi_gpu, finetuned_path = model_path)    
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    
    if args.do_eval:
        sample_exps = (not args.do_task and args.do_explain)
        print("\nGetting final eval results...\n")
        stats_dict = train_or_eval_epoch(args, device, dev_dataloader, stats_dict, multi_gpu = multi_gpu, 
                               model = model, 
                               optimizer = None, 
                               scheduler = None, 
                               tokenizer = tokenizer, 
                               sample_exps = sample_exps,
                               split_name = 'dev')   
        if data_name != 'QA' or args.labels_to_use != 'label':
            stats_dict = train_or_eval_epoch(args, device, test_dataloader, stats_dict, multi_gpu = multi_gpu, 
                               model = model, 
                               optimizer = None, 
                               scheduler = None, 
                               tokenizer = tokenizer,
                               sample_exps = sample_exps,
                               split_name = 'test')       
        
        # print and write final stats2
        dev_acc, test_acc, dev_bleu, test_bleu = -1, -1, -1, -1
        if data_name != 'QA' or args.labels_to_use != 'label': # CQA does not have test labels. so compute test on other datasets and CQA simulation
            if args.do_task:
                dev_acc = stats_dict['dev_acc']        
                test_acc = stats_dict['test_acc']
            if sample_exps:
                dev_bleu = stats_dict['dev_bleu']
                test_bleu = stats_dict['test_bleu']
            final_msg = f"Best epoch: {best_epoch} | Dev acc: {dev_acc:.2f} | Test acc: {test_acc:.2f} | Dev BLEU: {dev_bleu:.2f} | Test BLEU: {test_bleu:.2f}"
        else:
            if args.do_task: dev_acc = stats_dict['dev_acc']
            if sample_exps: dev_bleu = stats_dict['dev_bleu']
            final_msg = f"Best epoch: {best_epoch} | Dev acc: {dev_acc:.2f} | Dev BLEU: {dev_bleu:.2f}"    
        if args.do_train:
            report.write_final_score(args, final_score_str = final_msg, time_msg = time_msg)
        report.print_epoch_scores(epoch = best_epoch, scores = {k:v for k,v in stats_dict.items() if 'train' not in k})

    # write predictions
    if args.write_predictions:
        start_time = time.time()

        print("Writing preds for train...")
        stats_dict = train_or_eval_epoch(args, device, sequential_train_dataloader, stats_dict, multi_gpu = multi_gpu, 
                           model = model, 
                           optimizer = None, 
                           scheduler = None, 
                           tokenizer = tokenizer,
                           sample_exps = args.do_explain,
                           split_name = 'train',
                           write_predictions = True) 

        print("Writing preds for dev...")
        stats_dict = train_or_eval_epoch(args, device, dev_dataloader, stats_dict, multi_gpu = multi_gpu, 
                           model = model, 
                           optimizer = None, 
                           scheduler = None, 
                           tokenizer = tokenizer,
                           sample_exps = args.do_explain,
                           split_name = 'dev',
                           write_predictions = True) 

        print("Writing preds for test...")
        stats_dict = train_or_eval_epoch(args, device, test_dataloader, stats_dict, multi_gpu = multi_gpu, 
                           model = model, 
                           optimizer = None, 
                           scheduler = None, 
                           tokenizer = tokenizer,
                           sample_exps = args.do_explain,
                           split_name = 'test',
                           write_predictions = True) 

        end_time = time.time()
        writing_time = (end_time-start_time) / 60
        unit = 'minutes' if writing_time < 60 else 'hours'
        writing_time = writing_time if writing_time < 60 else writing_time / 60
        time_msg = f"\nTotal writing time: {writing_time:.2f} {unit}"
        print(time_msg)

        report.print_epoch_scores(epoch = -1, scores = stats_dict)

    ### END OF SCRIPT ###


