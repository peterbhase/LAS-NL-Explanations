import argparse
import os
import time

import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5Tokenizer

import t5_utils
import utils
from models.T5ForMC import T5ModelForMC
from t5_utils import T5Input, T5Output, print_t5_input, print_t5_output, make_t5_tensor
from utils import str2bool, Report


def load_model(args, base_model, device, tokenizer, finetuned_path=None):
    if finetuned_path is None:
        print(f"\nLoading non-finetuned model: {base_model}...")
    elif finetuned_path is not None:
        print(f"\nLoading fine-tuned model: {finetuned_path}...")
        if not os.path.exists(finetuned_path):
            raise ValueError('Saved model file does not exist.')

    model_class = T5ModelForMC

    model = model_class.from_pretrained(base_model, cache_dir=args.cache_dir)
    model.resize_token_embeddings(len(tokenizer))

    if finetuned_path is not None:
        model_state_dict = torch.load(finetuned_path, map_location=lambda storage,
                                                                          loc: storage)  # args for preventing memory leakage across gpus
        model.load_state_dict(model_state_dict)
        del model_state_dict

    model.to(device)

    return model


def prepare_optimizer(args, model, lr=None):
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    learning_rate = lr if lr else args.lr
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=learning_rate,
                      correct_bias=True)  # set to false to replicate BertAdam exactly

    return optimizer


def t5_qa_batch_forward(args, device, model, tokenizer, answer_input: T5Input):
    # forward passes
    encoder_outputs = model(input_ids=answer_input.encoder_inputs,
                            attention_mask=answer_input.encoder_masks)
    answer_encoder_hidden_states = encoder_outputs[1]
    if args.debug and args.verbose:
        print(f'answer_encoder_hidden_states: {answer_encoder_hidden_states}')

    decoder_outputs = model(encoder_hidden_states=answer_encoder_hidden_states,
                            encoder_attention_mask=answer_input.encoder_masks,
                            decoder_input_ids=answer_input.decoder_inputs,
                            decoder_lm_labels=answer_input.decoder_labels,
                            decoder_attention_mask=answer_input.decoder_masks,
                            batch_loss=True)

    qa_loss = decoder_outputs[0]

    return T5Output(answer_encoder_hidden_states, qa_loss, None)


def t5_qa_batch_predict(args, device, model, tokenizer, choices_input: T5Input, qa_output: T5Output):
    model.eval()
    # now get likelihoods for each choice
    if qa_output.choices_loss is None:
        with torch.no_grad():
            # add num_choices dim to input_masks and encoder_hidden_states and expand to match task_output_ids shape
            expand_shape = list(qa_output.encoder_hidden_states.shape)
            expand_shape.insert(1, choices_input.decoder_inputs.size(1))

            # expand hidden_states and input_masks.
            choices_encoder_hidden_states = qa_output.encoder_hidden_states.unsqueeze(1).expand(expand_shape)
            choices_input_masks = choices_input.encoder_masks.unsqueeze(1).expand_as(choices_input.decoder_inputs)

            decoder_outputs = model.QA_forward(encoder_hidden_states=choices_encoder_hidden_states,
                                               encoder_attention_mask=choices_input_masks,
                                               decoder_input_ids=choices_input.decoder_inputs,
                                               decoder_lm_labels=choices_input.decoder_labels,
                                               decoder_attention_mask=choices_input.decoder_masks)

            # choice_losses is of shape: batch_size x num_choices
            choices_loss = decoder_outputs[0]
            qa_output.choices_loss = choices_loss

    # compute task accuracy
    qa_labels = choices_input.choice_labels.detach().cpu().numpy()
    qa_predictions = np.argmin(qa_output.choices_loss.detach().cpu().numpy(), axis=-1)
    qa_acc_sum = np.sum(qa_predictions == qa_labels)

    qa_output.predictions = qa_predictions
    qa_output.acc_sum = qa_acc_sum


def t5_qa_batch_forward_ce(args, device, model, tokenizer, qa_choices_input: T5Input):
    # encode
    encoder_outputs = model(input_ids=qa_choices_input.encoder_inputs, attention_mask=qa_choices_input.encoder_masks)
    qa_encoder_hidden_states = encoder_outputs[1]

    # add num_choices dim to input_masks and encoder_hidden_states and expand to match qa_input.decoder_inputs shape
    expand_shape = list(qa_encoder_hidden_states.shape)
    expand_shape.insert(1, qa_choices_input.decoder_inputs.size(1))
    resized_encoder_hidden_states = qa_encoder_hidden_states.unsqueeze(1).expand(expand_shape)
    resized_encoder_masks = qa_choices_input.encoder_masks.unsqueeze(1).expand_as(qa_choices_input.decoder_inputs)

    decoder_outputs = model.QA_forward(encoder_hidden_states=resized_encoder_hidden_states,
                                       encoder_attention_mask=resized_encoder_masks,
                                       decoder_input_ids=qa_choices_input.decoder_inputs,
                                       decoder_lm_labels=qa_choices_input.decoder_labels,
                                       decoder_attention_mask=qa_choices_input.decoder_masks)

    choices_loss = decoder_outputs[0]  # (batch_size, num_choices)

    # ce loss
    choices_probs = torch.exp(-choices_loss)  # (batch_size, num_choices)
    choices_probs = choices_probs / torch.sum(choices_probs, dim=-1).unsqueeze(-1).expand_as(
        choices_probs)  # (batch_size, num_choices)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    qa_loss = loss_fct(choices_probs, qa_choices_input.choice_labels).unsqueeze(dim=-1)

    return T5Output(qa_encoder_hidden_states, qa_loss, None, choices_loss=choices_loss)


def t5_exp_batch_forward(args, device, model, tokenizer, exp_input: T5Input):
    # encode
    encoder_outputs = model(input_ids=exp_input.encoder_inputs, attention_mask=exp_input.encoder_masks)
    exp_encoder_hidden_states = encoder_outputs[1]

    decoder_outputs = model(encoder_hidden_states=exp_encoder_hidden_states,
                            encoder_attention_mask=exp_input.encoder_masks,
                            decoder_input_ids=exp_input.decoder_inputs,
                            decoder_lm_labels=exp_input.decoder_labels,
                            decoder_attention_mask=exp_input.decoder_masks,
                            batch_loss=True)
    exp_loss = decoder_outputs[0]

    return T5Output(exp_encoder_hidden_states, exp_loss, None, None)


def t5_exp_batch_sample(args, device, model, tokenizer, exp_input: T5Input, exp_output: T5Output,
                        choice_labels, sampling_strategy='argmax'):
    model.eval()
    with torch.no_grad():
        if choice_labels is None:
            raise ValueError('Need choice label if sample in forward pass.')
        correct_contexts = torch.stack([exp_input.context_ids[i, choice_labels[i], :] for i in
                                        range(exp_input.context_ids.size(0))], dim=0)
        if args.debug and args.verbose:
            print('correct_contexts: ', correct_contexts)

        samples = t5_utils.sample_batched(model,
                                          context_ids=correct_contexts.unsqueeze(1),
                                          tokenizer=tokenizer,
                                          max_sample_len=args.max_sample_len,
                                          model_name='T5',
                                          encoder_hidden_states=exp_output.encoder_hidden_states,
                                          input_masks=exp_input.encoder_masks,
                                          sampling_strategy=sampling_strategy)
        exp_predictions = samples.squeeze(1).tolist()
        exp_output.predictions = exp_predictions


def make_simulation_qa_inputs_dropout(args, device, tokenizer,
                                      task_qa_choices_input: T5Input, task_qa_output: T5Output,
                                      task_qa_encoder_x_masks, task_qa_encoder_e_masks,
                                      task_qa_encoder_input_strs, task_exp_prediction_strs):
    input_padding_id = tokenizer.pad_token_id
    label_padding_id = -100
    batch_size, max_seq_len = task_qa_choices_input.encoder_inputs.size()

    sim_qa_encoder_input_strs = []
    for idx in range(batch_size):
        task_qa_encoder_input_str = task_qa_encoder_input_strs[idx]
        explanation_str = f'My commonsense tells me {task_exp_prediction_strs[idx]}'
        sim_qa_encoder_input_strs.append(f'{task_qa_encoder_input_str} {explanation_str}')

    sim_qa_encoder_inputs, sim_qa_yxe_encoder_masks = make_t5_tensor(tokenizer, sim_qa_encoder_input_strs,
                                                                     input_padding_id, args.max_seq_len,
                                                                     add_eos=False, make_mask=True)

    sim_qa_yx_encoder_masks = sim_qa_yxe_encoder_masks.clone().cuda() * task_qa_encoder_x_masks
    sim_qa_ye_encoder_masks = sim_qa_yxe_encoder_masks.clone().cuda() * task_qa_encoder_e_masks

    if not args.ce_loss:
        assert False, 'Non ce_loss not supported'
    sim_qa_decoder_choices_inputs = task_qa_choices_input.decoder_inputs
    sim_qa_decoder_choices_masks = task_qa_choices_input.decoder_masks
    sim_qa_decoder_choices_labels = task_qa_choices_input.decoder_labels
    sim_qa_choice_labels = torch.tensor(task_qa_output.predictions, dtype=torch.long)
    sim_qa_yxe_choices_input = T5Input(sim_qa_encoder_inputs, sim_qa_yxe_encoder_masks, sim_qa_decoder_choices_inputs,
                                       sim_qa_decoder_choices_masks, sim_qa_decoder_choices_labels,
                                       choice_labels=sim_qa_choice_labels)
    sim_qa_yxe_choices_input.to_device(device)
    sim_qa_yx_choices_input = T5Input(sim_qa_encoder_inputs, sim_qa_yx_encoder_masks, sim_qa_decoder_choices_inputs,
                                      sim_qa_decoder_choices_masks, sim_qa_decoder_choices_labels,
                                      choice_labels=sim_qa_choice_labels)
    sim_qa_yx_choices_input.to_device(device)
    sim_qa_ye_choices_input = T5Input(sim_qa_encoder_inputs, sim_qa_ye_encoder_masks, sim_qa_decoder_choices_inputs,
                                      sim_qa_decoder_choices_masks, sim_qa_decoder_choices_labels,
                                      choice_labels=sim_qa_choice_labels)
    sim_qa_ye_choices_input.to_device(device)
    if args.explain_sim:
        assert False, 'Simulator language modeling objective not supported'

    return sim_qa_yxe_choices_input, sim_qa_yx_choices_input, sim_qa_ye_choices_input


def make_likelihood_inputs(args, device, tokenizer, task_exp_input: T5Input, task_exp_prediction_strs):
    input_padding_id = tokenizer.pad_token_id
    label_padding_id = -100
    batch_size, max_seq_len = task_exp_input.encoder_inputs.size()

    llh_exp_decoder_input_strs = [f'My commonsense tells me {task_exp_prediction_strs[idx]}' for idx in
                                  range(batch_size)]

    llh_exp_encoder_inputs, llh_exp_encoder_masks = task_exp_input.encoder_inputs, task_exp_input.encoder_masks
    llh_exp_decoder_inputs, llh_exp_decoder_masks = make_t5_tensor(tokenizer, llh_exp_decoder_input_strs,
                                                                   input_padding_id, args.max_seq_len,
                                                                   add_eos=False, make_mask=True)
    llh_exp_decoder_labels = make_t5_tensor(tokenizer, llh_exp_decoder_input_strs, label_padding_id,
                                            args.max_seq_len, add_eos=False, make_mask=False)
    llh_exp_input = T5Input(llh_exp_encoder_inputs, llh_exp_encoder_masks, llh_exp_decoder_inputs,
                            llh_exp_decoder_masks, llh_exp_decoder_labels)
    llh_exp_input.to_device(device)

    return llh_exp_input


def calculate_reward(args, sim_qa_input: T5Input, sim_qa_output: T5Output):
    with torch.no_grad():
        softmax_func = torch.nn.Softmax(dim=-1)
        choices_prob = softmax_func(-sim_qa_output.choices_loss.detach() / args.temperature)
        reward = torch.gather(choices_prob, dim=-1,
                              index=sim_qa_input.choice_labels.detach().view(-1, 1))
        if args.debug:
            print(f'choices_loss: {sim_qa_output.choices_loss.detach()}')
            print(f'choices_prob: {choices_prob}')
            print(f'reward: {reward}')
    return reward


def calculate_reward_baseline(args, sim_qa_yx_input: T5Input, sim_qa_yx_output: T5Output, sim_qa_ye_output: T5Output):
    with torch.no_grad():
        softmax_func = torch.nn.Softmax(dim=-1)
        yx_choices_prob = softmax_func(-sim_qa_yx_output.choices_loss.detach() / args.temperature)
        ye_choices_prob = softmax_func(-sim_qa_ye_output.choices_loss.detach() / args.temperature)
        more_confident = torch.gt(torch.max(yx_choices_prob, dim=1)[0],
                                  torch.max(ye_choices_prob, dim=1)[0]).float().view(-1, 1)
        choices_prob = more_confident * yx_choices_prob + \
                       (torch.ones_like(more_confident) - more_confident) * ye_choices_prob
        reward = torch.gather(choices_prob, dim=-1,
                              index=sim_qa_yx_input.choice_labels.detach().view(-1, 1))
        if args.debug:
            print(f'yx_choices_prob: {yx_choices_prob}')
            print(f'ye_choices_prob: {ye_choices_prob}')
            print(f'more confident: {more_confident}')
            print(f'choices_prob: {choices_prob}')
            print(f'reward: {reward}')
    return reward


def train_epoch(args, device, task_model, sim_model, tokenizer, task_optimizer, task_scheduler, sim_optimizer,
                sim_scheduler, dataloader, stats_dict):
    task_model.train()
    sim_model.train()
    # init stat vars
    task_qa_loss_sum = 0
    task_exp_loss_sum = 0
    task_llh_loss_sum = 0
    reward_sum = 0
    sim_qa_loss_sum = 0
    sim_exp_loss_sum = 0
    task_acc_sum = 0
    sim_acc_sum = 0
    n_steps = 0
    n_batches = len(dataloader)
    n_examples = len(dataloader.dataset)
    start_time = time.time()

    for step, batch in enumerate(dataloader):
        print(f"  Train | Step {step + 1} / {n_batches}", end='\r')

        # unpack batch variables
        batch = [item.to(device) for item in batch]
        task_qa_encoder_inputs, task_qa_encoder_masks, task_qa_encoder_x_masks, task_qa_encoder_e_masks, \
        task_qa_decoder_answer_inputs, task_qa_decoder_answer_masks, task_qa_decoder_answer_labels, \
        task_qa_decoder_choices_inputs, task_qa_decoder_choices_masks, task_qa_decoder_choices_labels, \
        task_qa_choice_labels, \
        task_exp_encoder_inputs, task_exp_encoder_masks, \
        task_exp_decoder_inputs, task_exp_decoder_masks, task_exp_decoder_labels, \
        task_exp_context_ids, task_exp_explanation_ids = batch

        batch_size = task_qa_encoder_inputs.size(0)

        task_qa_answer_input = T5Input(task_qa_encoder_inputs,
                                       task_qa_encoder_masks,
                                       task_qa_decoder_answer_inputs,
                                       task_qa_decoder_answer_masks,
                                       task_qa_decoder_answer_labels,
                                       choice_labels=task_qa_choice_labels)
        task_qa_choices_input = T5Input(task_qa_encoder_inputs,
                                        task_qa_encoder_masks,
                                        task_qa_decoder_choices_inputs,
                                        task_qa_decoder_choices_masks,
                                        task_qa_decoder_choices_labels,
                                        choice_labels=task_qa_choice_labels)
        task_exp_input = T5Input(task_exp_encoder_inputs,
                                 task_exp_encoder_masks,
                                 task_exp_decoder_inputs,
                                 task_exp_decoder_masks,
                                 task_exp_decoder_labels,
                                 context_ids=task_exp_context_ids,
                                 explanation_ids=task_exp_explanation_ids)

        if args.debug:
            print_t5_input(args, tokenizer, task_qa_answer_input, msg='task_qa_answer_input')
            print_t5_input(args, tokenizer, task_qa_choices_input, msg='task_qa_choices_input')
            print_t5_input(args, tokenizer, task_exp_input, msg='task_exp_input')

        # task qa forward pass
        task_qa_output = t5_qa_batch_forward(args, device, task_model, tokenizer, task_qa_answer_input)
        task_model.eval()
        t5_qa_batch_predict(args, device, task_model, tokenizer, task_qa_choices_input, task_qa_output)
        task_model.train()
        if args.debug:
            print_t5_output(args, tokenizer, task_qa_output, msg='task_qa_output')

        # task exp forward pass
        task_exp_output = None
        if args.explain_task or args.sample_task:
            task_exp_output = t5_exp_batch_forward(args, device, task_model, tokenizer, task_exp_input)
            if args.sample_task:
                sampling_strategy = args.rl_sampling_strategy if args.do_rl else 'argmax'
                task_model.eval()
                t5_exp_batch_sample(args, device, task_model, tokenizer, task_exp_input, task_exp_output,
                                    task_qa_choices_input.choice_labels,
                                    sampling_strategy=sampling_strategy)
                task_model.train()
            if args.debug:
                print_t5_output(args, tokenizer, task_exp_output, msg='task_exp_output')

        if args.do_rl:
            # sim forward pass
            # decode task inputs
            ignore_tokens_list = [tokenizer.pad_token, '[UNK]']

            task_qa_encoder_input_strs = utils.detok_batch(tokenizer, task_qa_choices_input.encoder_inputs,
                                                           ignore_tokens=ignore_tokens_list,
                                                           eos_token=tokenizer.eos_token)
            task_exp_prediction_strs = utils.detok_batch(tokenizer, task_exp_output.predictions,
                                                         ignore_tokens=ignore_tokens_list,
                                                         eos_token=tokenizer.eos_token)
            # prepare simulation input
            sim_qa_yxe_choices_input, \
            sim_qa_yx_choices_input, \
            sim_qa_ye_choices_input = make_simulation_qa_inputs_dropout(args, device,
                                                                        tokenizer,
                                                                        task_qa_choices_input,
                                                                        task_qa_output,
                                                                        task_qa_encoder_x_masks,
                                                                        task_qa_encoder_e_masks,
                                                                        task_qa_encoder_input_strs,
                                                                        task_exp_prediction_strs)
            if args.debug:
                print_t5_input(args, tokenizer, sim_qa_yxe_choices_input, msg='sim_qa_yxe_choices_input')
                print_t5_input(args, tokenizer, sim_qa_yx_choices_input, msg='sim_qa_yx_choices_input')
                print_t5_input(args, tokenizer, sim_qa_ye_choices_input, msg='sim_qa_ye_choices_input')

            # sim qa forward pass
            sim_qa_yxe_output = t5_qa_batch_forward_ce(args, device, sim_model, tokenizer, sim_qa_yxe_choices_input)
            sim_qa_yx_output = t5_qa_batch_forward_ce(args, device, sim_model, tokenizer, sim_qa_yx_choices_input)
            sim_qa_ye_output = t5_qa_batch_forward_ce(args, device, sim_model, tokenizer, sim_qa_ye_choices_input)
            sim_model.eval()
            t5_qa_batch_predict(args, device, sim_model, tokenizer, sim_qa_yxe_choices_input, sim_qa_yxe_output)
            sim_model.train()
            if args.debug:
                print_t5_output(args, tokenizer, sim_qa_yxe_output, msg='sim_qa_yxe_output')
                print_t5_output(args, tokenizer, sim_qa_yx_output, msg='sim_qa_yx_output')
                print_t5_output(args, tokenizer, sim_qa_ye_output, msg='sim_qa_ye_output')

            # prepare likelihood input
            llh_exp_input = make_likelihood_inputs(args, device, tokenizer, task_exp_input, task_exp_prediction_strs)
            if args.debug:
                print_t5_input(args, tokenizer, llh_exp_input, msg='llh_exp_input')

            # task model likelihood forward pass
            llh_exp_output = t5_exp_batch_forward(args, device, task_model, tokenizer, llh_exp_input)
            if args.debug:
                print_t5_output(args, tokenizer, llh_exp_output, msg='llh_exp_output')

            # calculate reward
            yxe_reward = calculate_reward(args, sim_qa_yxe_choices_input, sim_qa_yxe_output)
            ye_reward = calculate_reward(args, sim_qa_ye_choices_input, sim_qa_ye_output)
            reward = (1 - args.beta) * yxe_reward - args.beta * ye_reward
            if args.debug:
                print('yxe_reward: ', yxe_reward)
                print('ye_reward: ', ye_reward)

        # backprop
        loss = torch.zeros([batch_size, 1], device=device)
        if args.train_task:
            task_loss = task_qa_output.loss
            if args.explain_task:
                task_loss = args.task_coef * task_loss + (1 - args.task_coef) * task_exp_output.loss
            if args.do_rl:
                task_loss = (1 - args.alpha) * task_loss + args.alpha * reward * llh_exp_output.loss
            loss += task_loss
        if args.train_sim:
            if args.dataset == 'cqa':
                sim_loss = 0.5 * sim_qa_yxe_output.loss + 0.5 * sim_qa_yx_output.loss
            elif args.dataset == 'nli':
                sim_loss = 0.4 * sim_qa_yxe_output.loss + 0.4 * sim_qa_yx_output.loss + 0.2 * sim_qa_ye_output.loss
            loss += sim_loss

        loss = loss.mean() / args.grad_accumulation_factor
        if args.debug:
            print('loss: ', loss)

        loss.backward()

        # a few conditions here for stepping the optimizers and schedulers
        if (step + 1) % args.grad_accumulation_factor == 0:
            # step task
            torch.nn.utils.clip_grad_norm_(task_model.parameters(), args.max_grad_norm)
            task_optimizer.step()
            task_scheduler.step()
            task_optimizer.zero_grad()

            # step sim
            torch.nn.utils.clip_grad_norm_(sim_model.parameters(), args.max_grad_norm)
            sim_optimizer.step()
            sim_scheduler.step()
            sim_optimizer.zero_grad()

            n_steps += 1

        # track stats
        task_qa_loss_sum += task_qa_output.loss.sum().item()
        if args.explain_task or args.sample_task:
            task_exp_loss_sum += task_exp_output.loss.sum().item()
        task_acc_sum += task_qa_output.acc_sum
        if args.do_rl or args.train_sim:
            sim_qa_loss_sum += sim_qa_yxe_output.loss.sum().item()
            sim_acc_sum += sim_qa_yxe_output.acc_sum
        if args.do_rl:
            reward_sum += reward.sum().item()
            task_llh_loss_sum += llh_exp_output.loss.sum()

        # clean up
        del batch, loss

    # summary stats
    task_qa_loss_mean = task_qa_loss_sum / n_examples
    task_exp_loss_mean = task_exp_loss_sum / n_examples
    reward_mean = reward_sum / n_examples
    sim_qa_loss_mean = sim_qa_loss_sum / n_examples
    sim_exp_loss_mean = sim_exp_loss_sum / n_examples
    task_acc_mean = task_acc_sum / n_examples
    sim_acc_mean = sim_acc_sum / n_examples
    task_llh_loss_mean = task_llh_loss_sum / n_examples

    stats = {}
    stats.update({'train_task_qa_loss': task_qa_loss_mean,
                  'train_task_exp_loss': task_exp_loss_mean,
                  'train_reward': reward_mean,
                  'train_sim_qa_loss': sim_qa_loss_mean,
                  'train_sim_exp_loss': sim_exp_loss_mean,
                  'train_task_acc': task_acc_mean * 100,
                  'train_sim_acc': sim_acc_mean * 100,
                  'train_task_llh_loss': task_llh_loss_mean,
                  })
    stats_dict.update(stats)

    run_time = (time.time() - start_time) / 60
    print(f"\n  Train time: {run_time:1.2f} minutes")

    return stats_dict


def eval_epoch(args, device, task_model, sim_model, tokenizer, dataloader, stats_dict, is_test=False):
    task_model.eval()
    sim_model.eval()

    mode = 'Test' if is_test else 'Eval'

    # init stat vars
    task_acc_sum = 0
    task_acc_sum_x_only = 0
    task_acc_sum_e_only = 0
    task_leaked_sim_score_sum = 0
    task_not_leaked_sim_score_sum = 0
    sim_acc_sum = 0
    sim_accx_sum = 0
    sim_acce_sum = 0
    sim_leaked_score_sum = 0
    sim_not_leaked_score_sum = 0
    task_leak_count = 0
    sim_leak_count = 0
    n_batches = len(dataloader)
    n_examples = len(dataloader.dataset)
    start_time = time.time()
    all_truth_strs, all_task_sample_strs, all_sim_sample_strs = [], [], []
    all_task_preds = []

    if args.debug:
        print(f'\n----{mode} Epoch----\n')

    # ignore these tokens when decoding
    ignore_tokens_list = [tokenizer.pad_token, '[CLS]', '[SEP]' '[UNK]']

    for step, batch in enumerate(dataloader):
        print(f" {mode} | Step {step + 1} / {n_batches}", end='\r')

        # unpack batch variables
        batch = [item.to(device) for item in batch]
        task_qa_encoder_inputs, task_qa_encoder_masks, task_qa_encoder_x_masks, task_qa_encoder_e_masks, \
        task_qa_decoder_answer_inputs, task_qa_decoder_answer_masks, task_qa_decoder_answer_labels, \
        task_qa_decoder_choices_inputs, task_qa_decoder_choices_masks, task_qa_decoder_choices_labels, \
        task_qa_choice_labels, \
        task_exp_encoder_inputs, task_exp_encoder_masks, \
        task_exp_decoder_inputs, task_exp_decoder_masks, task_exp_decoder_labels, \
        task_exp_context_ids, task_exp_explanation_ids = batch

        task_qa_answer_input = T5Input(task_qa_encoder_inputs,
                                       task_qa_encoder_masks,
                                       task_qa_decoder_answer_inputs,
                                       task_qa_decoder_answer_masks,
                                       task_qa_decoder_answer_labels,
                                       choice_labels=task_qa_choice_labels)
        task_qa_choices_input = T5Input(task_qa_encoder_inputs,
                                        task_qa_encoder_masks,
                                        task_qa_decoder_choices_inputs,
                                        task_qa_decoder_choices_masks,
                                        task_qa_decoder_choices_labels,
                                        choice_labels=task_qa_choice_labels)
        task_exp_input = T5Input(task_exp_encoder_inputs,
                                 task_exp_encoder_masks,
                                 task_exp_decoder_inputs,
                                 task_exp_decoder_masks,
                                 task_exp_decoder_labels,
                                 context_ids=task_exp_context_ids,
                                 explanation_ids=task_exp_explanation_ids)
        if args.condition_on_explanation:
            task_qa_answer_input_x_only = T5Input(task_qa_encoder_inputs,
                                                  task_qa_encoder_x_masks,
                                                  task_qa_decoder_answer_inputs,
                                                  task_qa_decoder_answer_masks,
                                                  task_qa_decoder_answer_labels,
                                                  choice_labels=task_qa_choice_labels)
            task_qa_answer_input_e_only = T5Input(task_qa_encoder_inputs,
                                                  task_qa_encoder_e_masks,
                                                  task_qa_decoder_answer_inputs,
                                                  task_qa_decoder_answer_masks,
                                                  task_qa_decoder_answer_labels,
                                                  choice_labels=task_qa_choice_labels)
            task_qa_choices_input_x_only = T5Input(task_qa_encoder_inputs,
                                                   task_qa_encoder_x_masks,
                                                   task_qa_decoder_choices_inputs,
                                                   task_qa_decoder_choices_masks,
                                                   task_qa_decoder_choices_labels,
                                                   choice_labels=task_qa_choice_labels)
            task_qa_choices_input_e_only = T5Input(task_qa_encoder_inputs,
                                                   task_qa_encoder_e_masks,
                                                   task_qa_decoder_choices_inputs,
                                                   task_qa_decoder_choices_masks,
                                                   task_qa_decoder_choices_labels,
                                                   choice_labels=task_qa_choice_labels)
        if args.debug:
            print_t5_input(args, tokenizer, task_qa_answer_input, msg='task_qa_answer_input')
            print_t5_input(args, tokenizer, task_qa_choices_input, msg='task_qa_choices_input')
            print_t5_input(args, tokenizer, task_exp_input, msg='task_exp_input')
            if args.condition_on_explanation:
                print_t5_input(args, tokenizer, task_qa_answer_input_x_only, msg='task_qa_answer_input_x_only')
                print_t5_input(args, tokenizer, task_qa_choices_input_x_only, msg='task_qa_choices_input_x_only')
                print_t5_input(args, tokenizer, task_qa_answer_input_e_only, msg='task_qa_answer_input_e_only')
                print_t5_input(args, tokenizer, task_qa_choices_input_e_only, msg='task_qa_choices_input_e_only')

        with torch.no_grad():
            # task qa forward pass
            task_qa_output = t5_qa_batch_forward(args, device, task_model, tokenizer, task_qa_answer_input)
            t5_qa_batch_predict(args, device, task_model, tokenizer, task_qa_choices_input, task_qa_output)
            if args.debug:
                print_t5_output(args, tokenizer, task_qa_output, msg='task_qa_output')
            all_task_preds.extend(task_qa_output.predictions.tolist())

            if args.condition_on_explanation:
                task_qa_output_x_only = t5_qa_batch_forward(args, device, task_model, tokenizer,
                                                            task_qa_answer_input_x_only)
                task_qa_output_e_only = t5_qa_batch_forward(args, device, task_model, tokenizer,
                                                            task_qa_answer_input_e_only)
                t5_qa_batch_predict(args, device, task_model, tokenizer, task_qa_choices_input_x_only,
                                    task_qa_output_x_only)
                t5_qa_batch_predict(args, device, task_model, tokenizer, task_qa_choices_input_e_only,
                                    task_qa_output_e_only)
                if args.debug:
                    print_t5_output(args, tokenizer, task_qa_output_x_only, msg='task_qa_output_x_only')
                    print_t5_output(args, tokenizer, task_qa_output_e_only, msg='task_qa_output_e_only')

                # calculate sim score
                task_yxe = calculate_reward(args, task_qa_choices_input, task_qa_output)
                task_yx = calculate_reward(args, task_qa_choices_input_x_only, task_qa_output_x_only)
                # baseline = calculate_reward_baseline(args, task_qa_choices_input_x_only, task_qa_output_x_only,
                #                                      task_qa_output_e_only)
                task_leaked_label = (torch.tensor(task_qa_output_e_only.predictions, device=device)
                                     == task_qa_choices_input_e_only.choice_labels)
                task_sim_score = task_yxe - task_yx
                if args.debug:
                    print('task_leaked_label: ', task_leaked_label)
                    print('task_sim_score', task_sim_score)

            # task exp forward pass
            if args.explain_task or args.sample_task:
                task_exp_output = t5_exp_batch_forward(args, device, task_model, tokenizer, task_exp_input)
                t5_exp_batch_sample(args, device, task_model, tokenizer, task_exp_input, task_exp_output,
                                    task_qa_choices_input.choice_labels, sampling_strategy='argmax')
                if args.debug:
                    print_t5_output(args, tokenizer, task_exp_output, msg='task_exp_output')
                task_sample_strs = utils.detok_batch(tokenizer, task_exp_output.predictions,
                                                     ignore_tokens=ignore_tokens_list,
                                                     eos_token=tokenizer.eos_token)
                truth_label_strs = utils.detok_batch(tokenizer, task_exp_input.explanation_ids,
                                                     ignore_tokens=ignore_tokens_list,
                                                     eos_token=tokenizer.eos_token)
                if args.debug:
                    print(f'truth_strs: {truth_label_strs}')
                    print(f'task_sample_strs: {task_sample_strs}')
                all_truth_strs.extend(truth_label_strs)
                all_task_sample_strs.extend(task_sample_strs)

            if args.do_rl:
                # prepare simulation input
                ignore_tokens_list = [tokenizer.pad_token, '[UNK]']
                task_qa_encoder_input_strs = utils.detok_batch(tokenizer, task_qa_choices_input.encoder_inputs,
                                                               ignore_tokens=ignore_tokens_list,
                                                               eos_token=tokenizer.eos_token)
                task_exp_prediction_strs = utils.detok_batch(tokenizer, task_exp_output.predictions,
                                                             ignore_tokens=ignore_tokens_list,
                                                             eos_token=tokenizer.eos_token)
                sim_qa_yxe_choices_input, \
                sim_qa_yx_choices_input, \
                sim_qa_ye_choices_input = make_simulation_qa_inputs_dropout(args, device,
                                                                            tokenizer,
                                                                            task_qa_choices_input,
                                                                            task_qa_output,
                                                                            task_qa_encoder_x_masks,
                                                                            task_qa_encoder_e_masks,
                                                                            task_qa_encoder_input_strs,
                                                                            task_exp_prediction_strs)
                if args.debug:
                    print_t5_input(args, tokenizer, sim_qa_yxe_choices_input, msg='sim_qa_yxe_choices_input')
                    print_t5_input(args, tokenizer, sim_qa_yx_choices_input, msg='sim_qa_yx_choices_input')
                    print_t5_input(args, tokenizer, sim_qa_ye_choices_input, msg='sim_qa_ye_choices_input')

                # sim qa forward pass
                sim_qa_yxe_output = t5_qa_batch_forward_ce(args, device, sim_model, tokenizer, sim_qa_yxe_choices_input)
                sim_qa_yx_output = t5_qa_batch_forward_ce(args, device, sim_model, tokenizer, sim_qa_yx_choices_input)
                sim_qa_ye_output = t5_qa_batch_forward_ce(args, device, sim_model, tokenizer, sim_qa_ye_choices_input)
                t5_qa_batch_predict(args, device, sim_model, tokenizer, sim_qa_yxe_choices_input, sim_qa_yxe_output)
                t5_qa_batch_predict(args, device, sim_model, tokenizer, sim_qa_yx_choices_input, sim_qa_yx_output)
                t5_qa_batch_predict(args, device, sim_model, tokenizer, sim_qa_ye_choices_input, sim_qa_ye_output)
                if args.debug:
                    print_t5_output(args, tokenizer, sim_qa_yxe_output, msg='sim_qa_yxe_output')
                    print_t5_output(args, tokenizer, sim_qa_yx_output, msg='sim_qa_yx_output')
                    print_t5_output(args, tokenizer, sim_qa_ye_output, msg='sim_qa_ye_output')

                # calculate reward
                yxe_reward = calculate_reward(args, sim_qa_yxe_choices_input, sim_qa_yxe_output)
                yx_reward = calculate_reward(args, sim_qa_yx_choices_input, sim_qa_yx_output)
                # reward_baseline = calculate_reward_baseline(args, sim_qa_yx_choices_input, sim_qa_yx_output,
                #                                             sim_qa_ye_output)
                # reward = yxe_reward - reward_baseline
                sim_leaked_label = (torch.tensor(sim_qa_ye_output.predictions, device=device)
                                    == sim_qa_ye_choices_input.choice_labels)
                sim_score = yxe_reward - yx_reward
                if args.debug:
                    print('sim_leaked_label: ', sim_leaked_label)
                    print('sim_score: ', sim_score)

        # track stats
        task_acc_sum += task_qa_output.acc_sum
        if args.condition_on_explanation:
            task_acc_sum_x_only += task_qa_output_x_only.acc_sum
            task_acc_sum_e_only += task_qa_output_e_only.acc_sum
            task_leaked_sim_score_sum += (task_sim_score * task_leaked_label).sum().item()
            task_not_leaked_sim_score_sum += (task_sim_score * torch.logical_not(task_leaked_label)).sum().item()
            task_leak_count += task_leaked_label.sum().item()
        if args.do_rl or args.train_sim:
            sim_acc_sum += sim_qa_yxe_output.acc_sum
            sim_accx_sum += sim_qa_yx_output.acc_sum
            sim_acce_sum += sim_qa_ye_output.acc_sum
            sim_leaked_score_sum += (sim_score * sim_leaked_label).sum().item()
            sim_not_leaked_score_sum += (
                    sim_score * torch.logical_not(sim_leaked_label)).sum().item()
            sim_leak_count += sim_leaked_label.sum().item()

        # clean up from batch
        del batch

    # summary stats
    stats = {}
    task_acc_mean = task_acc_sum / n_examples
    task_acc_mean_x_only = task_acc_sum_x_only / n_examples
    task_acc_mean_e_only = task_acc_sum_e_only / n_examples
    task_sim_score_mean = 0
    if task_leak_count != 0:
        task_sim_score_mean = (task_leaked_sim_score_sum / task_leak_count + task_not_leaked_sim_score_sum / (
                n_examples - task_leak_count)) / 2
    sim_acc_mean = sim_acc_sum / n_examples
    sim_accx_mean = sim_accx_sum / n_examples
    sim_acce_mean = sim_acce_sum / n_examples
    sim_score_mean = 0
    if sim_leak_count != 0:
        sim_score_mean = (sim_leaked_score_sum / sim_leak_count + sim_not_leaked_score_sum / (
                n_examples - sim_leak_count)) / 2
    if args.debug:
        print('task_leak_count: ', task_leak_count)
        print('sim_leak_count: ', sim_leak_count)

    # task bleu
    task_bleu = 0
    if args.explain_task or args.sample_task:
        task_bleu = utils.compute_bleu(all_task_sample_strs, [[x] for x in all_truth_strs])

    # sim bleu
    sim_bleu = 0
    # if args.explain_sim:
    #     sim_bleu = utils.compute_bleu(all_sim_sample_strs, [[x] for x in all_truth_strs])

    if is_test:
        stats.update({'test_task_acc': task_acc_mean * 100,
                      'test_task_acc_x_only': task_acc_mean_x_only * 100,
                      'test_task_acc_e_only': task_acc_mean_e_only * 100,
                      'test_task_sim_score': task_sim_score_mean * 100,
                      'test_task_bleu': task_bleu,
                      'test_sim_acc': sim_acc_mean * 100,
                      'test_sim_accx': sim_accx_mean * 100,
                      'test_sim_acce': sim_acce_mean * 100,
                      'test_sim_score': sim_score_mean * 100,
                      'test_sim_bleu': sim_bleu})
    else:
        stats.update({'eval_task_acc': task_acc_mean * 100,
                      'eval_task_acc_x_only': task_acc_mean_x_only * 100,
                      'eval_task_acc_e_only': task_acc_mean_e_only * 100,
                      'eval_task_sim_score': task_sim_score_mean * 100,
                      'eval_task_bleu': task_bleu,
                      'eval_sim_acc': sim_acc_mean * 100,
                      'eval_sim_accx': sim_accx_mean * 100,
                      'eval_sim_acce': sim_acce_mean * 100,
                      'eval_sim_score': sim_score_mean * 100,
                      'eval_sim_bleu': sim_bleu})
    stats_dict.update(stats)

    run_time = (time.time() - start_time) / 60
    print(f"\n  Eval time: {run_time:.2f} minutes")

    return stats_dict


def predict_epoch(args, device, task_model, sim_model, tokenizer, dataloader, data_file, output_file):
    task_model.eval()
    sim_model.eval()

    # init stat vars
    n_batches = len(dataloader)
    start_time = time.time()
    all_truth_strs, all_task_sample_strs, all_sim_sample_strs = [], [], []
    all_task_preds = []

    if args.debug:
        print(f'\n----Predict Epoch----\n')

    # ignore these tokens when decoding
    ignore_tokens_list = [tokenizer.pad_token, '[CLS]', '[SEP]' '[UNK]']

    for step, batch in enumerate(dataloader):
        print(f" Predict | Step {step + 1} / {n_batches}", end='\r')

        # unpack batch variables
        batch = [item.to(device) for item in batch]
        task_qa_encoder_inputs, task_qa_encoder_masks, task_qa_encoder_x_masks, task_qa_encoder_e_masks, \
        task_qa_decoder_answer_inputs, task_qa_decoder_answer_masks, task_qa_decoder_answer_labels, \
        task_qa_decoder_choices_inputs, task_qa_decoder_choices_masks, task_qa_decoder_choices_labels, \
        task_qa_choice_labels, \
        task_exp_encoder_inputs, task_exp_encoder_masks, \
        task_exp_decoder_inputs, task_exp_decoder_masks, task_exp_decoder_labels, \
        task_exp_context_ids, task_exp_explanation_ids = batch

        task_qa_answer_input = T5Input(task_qa_encoder_inputs,
                                       task_qa_encoder_masks,
                                       task_qa_decoder_answer_inputs,
                                       task_qa_decoder_answer_masks,
                                       task_qa_decoder_answer_labels,
                                       choice_labels=task_qa_choice_labels)
        task_qa_choices_input = T5Input(task_qa_encoder_inputs,
                                        task_qa_encoder_masks,
                                        task_qa_decoder_choices_inputs,
                                        task_qa_decoder_choices_masks,
                                        task_qa_decoder_choices_labels,
                                        choice_labels=task_qa_choice_labels)
        task_exp_input = T5Input(task_exp_encoder_inputs,
                                 task_exp_encoder_masks,
                                 task_exp_decoder_inputs,
                                 task_exp_decoder_masks,
                                 task_exp_decoder_labels,
                                 context_ids=task_exp_context_ids,
                                 explanation_ids=task_exp_explanation_ids)
        if args.debug:
            print_t5_input(args, tokenizer, task_qa_answer_input, msg='task_qa_answer_input')
            print_t5_input(args, tokenizer, task_qa_choices_input, msg='task_qa_choices_input')
            print_t5_input(args, tokenizer, task_exp_input, msg='task_exp_input')

        with torch.no_grad():
            # task qa forward pass
            task_qa_output = t5_qa_batch_forward(args, device, task_model, tokenizer, task_qa_answer_input)
            t5_qa_batch_predict(args, device, task_model, tokenizer, task_qa_choices_input, task_qa_output)
            if args.debug:
                print_t5_output(args, tokenizer, task_qa_output, msg='task_qa_output')
            all_task_preds.extend(task_qa_output.predictions.tolist())

            # task exp forward pass
            if args.explain_task or args.sample_task:
                task_exp_output = t5_exp_batch_forward(args, device, task_model, tokenizer, task_exp_input)
                t5_exp_batch_sample(args, device, task_model, tokenizer, task_exp_input, task_exp_output,
                                    task_qa_choices_input.choice_labels, sampling_strategy='argmax')
                if args.debug:
                    print_t5_output(args, tokenizer, task_exp_output, msg='task_exp_output')
                task_sample_strs = utils.detok_batch(tokenizer, task_exp_output.predictions,
                                                     ignore_tokens=ignore_tokens_list,
                                                     eos_token=tokenizer.eos_token)
                if args.debug:
                    print(f'task_sample_strs: {task_sample_strs}')
                all_task_sample_strs.extend(task_sample_strs)

        # clean up from batch
        del batch

    run_time = (time.time() - start_time) / 60
    print(f"\n  Predict time: {run_time:.2f} minutes")

    # write predictions
    if args.debug:
        print('')
        print('task predictions: ', all_task_preds)
        if args.explain_task or args.sample_task:
            print('task explanations: ', all_task_sample_strs)

    if args.dataset == 'sm':
        write_func = utils.write_prediction_to_sm_file
    elif args.dataset == 'cqa':
        write_func = utils.write_prediction_to_cqa_file
    elif args.dataset == 'nli':
        write_func = utils.write_prediction_to_nli_file
    pred_dict = {'t5_prediction': all_task_preds}
    if args.explain_task or args.sample_task:
        pred_dict['t5_explanation'] = all_task_sample_strs
    write_func(pred_dict, data_file, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model args
    parser.add_argument("--task_base_model", default='t5-base', type=str)
    parser.add_argument("--sim_base_model", default='t5-small', type=str)
    parser.add_argument("--max_seq_len", default=175, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--model_name', type=str, default='',
                        help="Save and/or load name for model. Generated training report will use model_name as suffix.")
    # hyperparams
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training. "
                             "Effective batch size is train_batch_size times grad_accumulation_factor.")
    parser.add_argument('--grad_accumulation_factor', type=int, default=3,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_batch_size", default=4, type=int, help="Batch size for evaluation.")
    parser.add_argument("--task_lr", default=1e-5, type=float, help="Initial learning rate for task.")
    parser.add_argument("--sim_lr", default=1e-4, type=float, help="Initial learning rate for simulator")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. ")
    parser.add_argument('--max_grad_norm', type=int, default=1, help="Maximum allowed norm for backprop gradients.")
    parser.add_argument("--task_coef", default=0.5, type=float, help="Coefficient for mixing task and explain loss.")
    parser.add_argument('--max_sample_len', type=int, default=20,
                        help='Maximum num tokens that can appear in generated explanation')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='Mixing ratio between normal training loss and rl loss.')
    parser.add_argument('--temperature', type=float, default=1,
                        help='Temperature for reward softmax. Higher temperature amplifies reward signal.')
    parser.add_argument('--patience', type=int, default=5,
                        help='Training stops after patience number of epochs without increase in accuracy on '
                             'evaluation set.')
    parser.add_argument('--mu', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.2, help='Mixing ratio between y|x,e reward and y|e reward.')
    # gpu + misc
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id to use')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    # directories + file paths
    parser.add_argument("--save_dir", default='saved_models/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--data_dir', type=str, default='training_reports/',
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--report_dir", default='training_reports/', type=str,
                        help="The output directory where the model training reports will be written.")
    parser.add_argument("--cache_dir", default='cache/', type=str,
                        help="Directory for cacheing pretrained models.")
    parser.add_argument('--train_data_file', type=str, default='train.csv')
    parser.add_argument('--eval_data_file', type=str, default='dev.csv')
    parser.add_argument('--test_data_file', type=str, default='test.csv')
    parser.add_argument('--train_output_file', type=str, default='',
                        help='Output file to write training set predictions to.')
    parser.add_argument('--eval_output_file', type=str, default='',
                        help='Output file to write evaluation set predictions to.')
    parser.add_argument('--test_output_file', type=str, default='',
                        help='Output file to write test set predictions to.')
    # debug flags
    parser.add_argument('--small_data', type=int, default=0,
                        help='Flag for using just a few datapoints for debugging purposes')
    parser.add_argument('--debug', action='store_true', help='Flag that prints a lot of things out.')
    parser.add_argument('--verbose', action='store_true', help='Print even more stuff out.')
    # experiment condition flags
    parser.add_argument("--explanation_only", default=False, help="If set to true, only use explanation in qa input.")
    parser.add_argument("--condition_on_explanation", default=False,
                        help="Whether or not to condition on explanations in input")
    parser.add_argument("--explanation_to_use", type=str, default='truth', choices=['truth', 't5'])
    parser.add_argument("--label_to_use", type=str, default='truth', choices=['truth', 't5'])
    parser.add_argument('--explain_task', default=True, type=str2bool, help='Whether to use LM for task model')
    parser.add_argument('--sample_task', default=True, type=str2bool, help='Whether to sample agent 1')
    parser.add_argument('--train_task', default=True, type=str2bool, help='Whether to train agent 1')
    parser.add_argument('--train_sim', default=True, type=str2bool, help='Whether to train agent 2')
    parser.add_argument('--explain_sim', default=False, type=str2bool, help='Whether to use LM for simulation model')
    parser.add_argument('--do_rl', default=True, type=str2bool, help='Whether to use harry reward term in agent 1 loss')
    parser.add_argument('--rl_sampling_strategy', type=str, default='multinomial', choices=['multinomial', 'argmax'],
                        help='Sampling strategy for agent 1 during harry. If do_rl=False, it is set to argmax')
    parser.add_argument('--ce_loss', default=True, type=str2bool)
    parser.add_argument('--rationalize', default=False, type=str2bool)
    # control flow for script
    parser.add_argument("--do_train", default=True, type=str2bool, help="Whether to run training.")
    parser.add_argument('--do_test', default=False, type=str2bool)
    parser.add_argument("--sequential_train", type=str2bool, default=False)
    parser.add_argument('--pretrained_task', type=str, default='',
                        help="Load task model from saved model file")
    parser.add_argument('--pretrained_sim', type=str, default='',
                        help="Load simulation model from saved model file")
    parser.add_argument('--select_for', type=str, default='sim_acc', choices=['task_acc', 'sim_acc'])
    parser.add_argument('--write_prediction', action='store_true', help='Whether to write agent 1 predictions to file')
    parser.add_argument('--dataset', type=str, default='sm', choices=['sm', 'cqa', 'nli'])
    parser.add_argument('--save_all_epochs', type=str2bool, default=False)

    args = parser.parse_args()
    if args.do_rl:
        assert args.sample_task and args.train_task, 'If using rl reward, need to train and sample from agent 1'
    if not (args.train_task or args.train_sim):
        assert not args.do_train, 'Either train_task or train_sim needs to be true if training'
    if args.sample_task:
        assert args.do_rl or args.train_sim, 'Wasteful sampling'
    if args.condition_on_explanation or args.explanation_only:
        assert not args.explain_task
    if args.explanation_only:
        assert not args.condition_on_explanation

    # gpu setup
    n_gpu = torch.cuda.device_count()
    multi_gpu = (n_gpu > 1 and args.gpu == -1)  # i.e. multiple gpus available and gpu choice not specified
    if multi_gpu:
        device = torch.device("cuda") if args.gpu == -1 else torch.device(f'cuda:{args.gpu}')
        assert args.train_batch_size % n_gpu == 0, f"Train batch size will need to be allocated equally across {n_gpu} gpus, but {args.train_batch_size} cannot be"
        assert args.eval_batch_size % n_gpu == 0, f"Eval batch size will need to be allocated equally across {n_gpu} gpus, but {args.eval_batch_size} cannot be"
    elif args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed setup
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if multi_gpu:
        torch.cuda.manual_seed_all(args.seed)
    if args.debug:
        print(f'seed: {args.seed}')

    # debug setting
    if args.debug:
        args.model_name = 'debug'
        args.small_data = 1 if args.small_data == 0 else args.small_data
        args.train_batch_size = args.small_data
        args.eval_batch_size = args.small_data
        args.num_train_epochs = 1
        args.sequential_train = True
        args.grad_accumulation_factor = 1

    # make paths and dirs
    task_model_file = os.path.join(args.save_dir, f"{args.model_name}_task.hdf5")
    sim_model_file = os.path.join(args.save_dir, f"{args.model_name}_sim.hdf5")

    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)
    if not os.path.exists(args.report_dir): os.makedirs(args.report_dir)
    if not args.train_output_file: args.train_output_file = args.train_data_file
    if not args.eval_output_file: args.eval_output_file = args.eval_data_file
    if not args.test_output_file: args.test_output_file = args.test_data_file
    if args.data_dir:
        args.train_data_file = os.path.join(args.data_dir, args.train_data_file)
        args.eval_data_file = os.path.join(args.data_dir, args.eval_data_file)
        args.test_data_file = os.path.join(args.data_dir, args.test_data_file)
    if args.report_dir:
        args.train_output_file = os.path.join(args.report_dir, args.train_output_file)
        args.eval_output_file = os.path.join(args.report_dir, args.eval_output_file)
        args.test_output_file = os.path.join(args.report_dir, args.test_output_file)
    args.pretrained_task = os.path.join(args.save_dir, args.pretrained_task) if args.pretrained_task else None
    args.pretrained_sim = os.path.join(args.save_dir, args.pretrained_sim) if args.pretrained_sim else None

    # make Report and stats_dict
    stats_dict = {}

    # load tokenizer. note T5 tokenizer had pad and eos tokens by default
    tokenizer = T5Tokenizer.from_pretrained(args.task_base_model, cache_dir=args.cache_dir)

    # load data
    train_dataloader, eval_dataloader, test_dataloader = t5_utils.make_t5_dataloader(args, tokenizer,
                                                                                     sequential=args.sequential_train,
                                                                                     do_test=args.do_test)
    print(f"Data set sizes:\n"
          f"  Train: {len(train_dataloader.dataset)}\n"
          f"  Eval: {len(eval_dataloader.dataset)}")
    if args.do_test:
        print(f"  Test: {len(test_dataloader.dataset)}")
    print('')

    # begin training
    best_epoch = -1.0
    best_score = -1.0
    if args.do_train:

        # create report
        report_name = f"report_{args.model_name}.txt"
        report_file = os.path.join(args.report_dir, report_name)
        score_names = ['train_task_qa_loss', 'train_task_exp_loss', 'train_sim_qa_loss', 'train_sim_exp_loss',
                       'train_task_acc', 'train_sim_acc', 'train_reward', 'train_task_llh_loss',
                       'eval_task_acc', 'eval_task_acc_x_only', 'eval_task_acc_e_only',
                       'eval_task_sim_score', 'eval_task_bleu',
                       'eval_sim_acc', 'eval_sim_accx', 'eval_sim_acce', 'eval_sim_score', 'eval_sim_bleu']
        report = Report(args, report_file, score_names=score_names)

        # load models
        task_model = load_model(args, args.task_base_model, device, tokenizer, finetuned_path=args.pretrained_task)
        sim_model = load_model(args, args.sim_base_model, device, tokenizer, finetuned_path=args.pretrained_sim)

        # load optimizer
        num_train_optimization_steps = args.num_train_epochs * int(
            len(train_dataloader.dataset) / args.train_batch_size / args.grad_accumulation_factor)
        task_optimizer = prepare_optimizer(args, model=task_model, lr=args.task_lr)
        sim_optimizer = prepare_optimizer(args, model=sim_model, lr=args.sim_lr)

        # make scheduler
        task_scheduler = get_linear_schedule_with_warmup(task_optimizer,
                                                         num_warmup_steps=int(
                                                             args.warmup_proportion * num_train_optimization_steps),
                                                         num_training_steps=num_train_optimization_steps)
        sim_scheduler = get_linear_schedule_with_warmup(sim_optimizer,
                                                        num_warmup_steps=int(
                                                            args.warmup_proportion * num_train_optimization_steps),
                                                        num_training_steps=num_train_optimization_steps)

        # training loop
        print("\nBeginning training...\n")
        start_time = time.time()
        patience_count = 0
        for e in range(args.num_train_epochs):
            print(f"Epoch {e}")
            print(f'LR: {sim_optimizer.param_groups[0]["lr"]}')

            stats_dict = train_epoch(args, device, task_model, sim_model, tokenizer, task_optimizer,
                                     task_scheduler, sim_optimizer, sim_scheduler, train_dataloader, stats_dict)
            stats_dict = eval_epoch(args, device, task_model, sim_model, tokenizer, eval_dataloader, stats_dict)

            score = stats_dict[f'eval_{args.select_for}']
            # check for best dev score and save if new best
            task_model_save = task_model.module if hasattr(task_model, 'module') else task_model
            sim_model_save = sim_model.moduel if hasattr(sim_model, 'module') else sim_model
            if args.save_all_epochs:
                torch.save(task_model_save.state_dict(), os.path.join(task_model_file, f'epoch{e}'))
                torch.save(sim_model_save.state_dict(), os.path.join(sim_model_file, f'epoch{e}'))
            elif score > best_score:
                print(f"  New best model. Saving model(s) in {args.save_dir}")
                torch.save(task_model_save.state_dict(), task_model_file)
                torch.save(sim_model_save.state_dict(), sim_model_file)
                best_score = score
                best_epoch = e
                patience_count = 0
            else:
                patience_count += 1

            # write + print summary stats
            report.write_epoch_scores(epoch=e, scores=stats_dict)
            utils.print_epoch_scores(epoch=e, scores=stats_dict)

            if patience_count >= args.patience:
                break

        end_time = time.time()
        training_time = (end_time - start_time) / 60
        unit = 'minutes' if training_time < 60 else 'hours'
        training_time = training_time if training_time < 60 else training_time / 60
        print(f"\nTotal training time: {training_time:.2f} {unit}")

    # final eval
    final_task_model_file = task_model_file if args.do_train else args.pretrained_task
    final_sim_model_file = sim_model_file if args.do_train else args.pretrained_sim
    task_model = load_model(args, args.task_base_model, device, tokenizer, finetuned_path=final_task_model_file)
    sim_model = load_model(args, args.sim_base_model, device, tokenizer, finetuned_path=final_sim_model_file)

    print("\nGetting final eval results...\n")
    stats_dict = eval_epoch(args, device, task_model, sim_model, tokenizer, eval_dataloader, stats_dict)
    eval_task_acc = stats_dict['eval_task_acc']
    eval_sim_acc = stats_dict['eval_sim_acc']
    eval_task_bleu = stats_dict['eval_task_bleu']
    eval_sim_bleu = stats_dict['eval_sim_bleu']
    utils.print_epoch_scores(epoch=best_epoch, scores={k: v for k, v in stats_dict.items() if 'eval' in k})

    # test
    if args.do_test:
        print("\nGetting final test results...\n")
        stats_dict = eval_epoch(args, device, task_model, sim_model, tokenizer, test_dataloader, stats_dict,
                                is_test=True)
        test_task_acc = stats_dict['test_task_acc']
        test_sim_acc = stats_dict['test_sim_acc']
        test_task_bleu = stats_dict['test_task_bleu']
        test_sim_bleu = stats_dict['test_sim_bleu']
        utils.print_epoch_scores(epoch=best_epoch, scores={k: v for k, v in stats_dict.items() if 'test' in k})

    # write final message to report
    if args.do_train:
        final_msg = f"Final results.\n" \
                    f"Best epoch: {best_epoch}\n" \
                    f"Best eval task acc: {eval_task_acc:.2f}\n" \
                    f"Best eval sim acc: {eval_sim_acc:.2f}\n" \
                    f"Best eval task bleu: {eval_task_bleu:.2f}\n" \
                    f"Best eval sim bleu: {eval_sim_bleu:.2f}\n"
        if args.do_test:
            final_msg += f"Test task acc: {test_task_acc:.2f}\n" \
                         f"Test sim acc: {test_sim_acc:.2f}\n" \
                         f"Test task bleu: {test_task_bleu:.2f}\n" \
                         f"Test sim bleu: {test_sim_bleu:.2f}\n"
        report.write_final_score(args, final_score_str=final_msg)

    if args.write_prediction:
        # load sequential data for prediction
        train_dataloader, eval_dataloader, test_dataloader = t5_utils.make_t5_dataloader(args, tokenizer,
                                                                                         sequential=True, do_test=True)

        print("Writing preds for train...")
        predict_epoch(args, device, task_model, sim_model, tokenizer, train_dataloader,
                      data_file=args.train_data_file,
                      output_file=args.train_output_file)

        print("Writing preds for eval...")
        predict_epoch(args, device, task_model, sim_model, tokenizer, eval_dataloader,
                      data_file=args.eval_data_file,
                      output_file=args.eval_output_file)

        print("Writing preds for test...")
        predict_epoch(args, device, task_model, sim_model, tokenizer, test_dataloader,
                      data_file=args.test_data_file,
                      output_file=args.test_output_file)

    print('\n----Done----\n')
