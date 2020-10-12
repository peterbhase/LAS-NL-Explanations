import torch
from typing import Sequence

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import utils
from utils import CQAExample, SMExample, NLIExample
from utils import truncate_seq_pair, detok_batch


class T5Input:

    def __init__(self, encoder_inputs, encoder_masks, decoder_inputs, decoder_masks, decoder_labels, choice_labels=None,
                 context_ids=None, explanation_ids=None):
        self.encoder_inputs = encoder_inputs
        self.encoder_masks = encoder_masks
        self.decoder_inputs = decoder_inputs
        self.decoder_masks = decoder_masks
        self.decoder_labels = decoder_labels
        self.choice_labels = choice_labels
        self.context_ids = context_ids
        self.explanation_ids = explanation_ids

    def to_device(self, device):
        for attr, value in self.__dict__.items():
            if value is not None:
                self.__dict__[attr] = value.to(device)


class T5Output:

    def __init__(self, encoder_hidden_states, loss, decoder_logits, predictions=None, acc_sum=None, bleu=None,
                 choices_loss=None):
        self.encoder_hidden_states = encoder_hidden_states
        self.loss = loss
        self.decoder_logits = decoder_logits
        self.predictions = predictions
        self.acc_sum = acc_sum
        self.bleu = bleu
        self.choices_loss = choices_loss


def make_t5_dataloader(args, tokenizer, sequential, do_test):
    if args.dataset == 'sm':
        read_func = utils.read_sm_examples
        make_input_func = utils.read_sm_examples
    elif args.dataset == 'cqa':
        read_func = utils.read_cqa_examples
        make_input_func = make_t5_cqa_inputs
    elif args.dataset == 'nli':
        read_func = utils.read_nli_examples
        make_input_func = make_t5_nli_inputs

    train_examples = read_func(args.train_data_file)
    eval_examples = read_func(args.eval_data_file)

    # small data for debugging purposes
    if args.small_data > 0:
        train_examples = train_examples[:args.small_data]
        eval_examples = eval_examples[:args.small_data]

    # convert examples to lists of tensors, and put into TensorDatasets then dataloaders.
    # use_explanations is flag for excluded explanations in inputs
    train_tensors = make_input_func(args, tokenizer, train_examples)
    train_data = TensorDataset(*train_tensors)
    train_sampler = RandomSampler(train_data) if not sequential else SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                  num_workers=4, pin_memory=True)

    eval_tensors = make_input_func(args, tokenizer, eval_examples)
    eval_data = TensorDataset(*eval_tensors)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)

    test_dataloader = None
    if do_test:
        test_examples = read_func(args.test_data_file)
        if args.small_data > 0:
            test_examples = test_examples[:args.small_data]
        test_tensors = make_input_func(args, tokenizer, test_examples)
        test_data = TensorDataset(*test_tensors)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)

    return train_dataloader, eval_dataloader, test_dataloader


def make_t5_sm_inputs(args, tokenizer, examples):
    qa_encoder_input_strs = []
    qa_decoder_answer_input_strs = []
    qa_decoder_answer_label_strs = []
    qa_decoder_choices_input_strs = []
    qa_decoder_choices_label_strs = []
    exp_encoder_input_strs = []
    exp_decoder_input_strs = []
    exp_decoder_label_strs = []
    exp_context_strs = []
    exp_explanation_strs = []

    for idx, example in enumerate(examples):
        qa_prefix = 'task: '
        exp_prefix = 'explain: '
        question_str = f'{example.statements[0]} [SEP] {example.statements[1]}'

        if args.label_to_use == 't5':
            answer_str = example.statements[int(example.input_dict['t5_prediction'])]
        else:
            answer_str = example.statements[example.statement_label]

        if args.explanation_to_use == 't5':
            explanation_str = example.input_dict['t5_explanation']
        else:
            explanation_str = example.human_explanation

        if not args.condition_on_explanation:
            qa_input_str = f'[CLS] {question_str} [SEP]'
        else:
            qa_input_str = f'[CLS] {question_str} [SEP] {explanation_str}'
        exp_input_str = f'[CLS] {question_str} [SEP]'

        qa_encoder_input_str = qa_prefix + qa_input_str
        qa_decoder_answer_input_str = f'The answer is: {answer_str}'
        qa_decoder_answer_label_str = qa_decoder_answer_input_str
        qa_decoder_choices_input_str = [f'The answer is: {statement}' for statement in example.statements]
        qa_decoder_choices_label_str = qa_decoder_choices_input_str

        exp_encoder_input_str = exp_prefix + exp_input_str
        exp_decoder_input_str = f'My common sense tells me {explanation_str}'
        exp_decoder_label_str = exp_decoder_input_str
        exp_context_str = ['My common sense tells me ' for statement in example.statements]
        exp_explanation_str = explanation_str

        qa_encoder_input_strs.append(qa_encoder_input_str)
        qa_decoder_answer_input_strs.append(qa_decoder_answer_input_str)
        qa_decoder_answer_label_strs.append(qa_decoder_answer_label_str)
        qa_decoder_choices_input_strs.append(qa_decoder_choices_input_str)
        qa_decoder_choices_label_strs.append(qa_decoder_choices_label_str)
        exp_encoder_input_strs.append(exp_encoder_input_str)
        exp_decoder_input_strs.append(exp_decoder_input_str)
        exp_decoder_label_strs.append(exp_decoder_label_str)
        exp_context_strs.append(exp_context_str)
        exp_explanation_strs.append(exp_explanation_str)

    input_padding_id = tokenizer.pad_token_id
    label_padding_id = -100
    qa_encoder_inputs, qa_encoder_masks = make_t5_tensor(tokenizer, qa_encoder_input_strs, input_padding_id,
                                                         args.max_seq_len, add_eos=False, make_mask=True)
    qa_decoder_answer_inputs, qa_decoder_answer_masks = make_t5_tensor(tokenizer, qa_decoder_answer_input_strs,
                                                                       input_padding_id,
                                                                       args.max_seq_len, add_eos=False,
                                                                       make_mask=True)
    qa_decoder_answer_labels = make_t5_tensor(tokenizer, qa_decoder_answer_label_strs, label_padding_id,
                                              args.max_seq_len, add_eos=False, make_mask=False)
    qa_decoder_choices_inputs, qa_decoder_choices_masks = make_t5_tensor(tokenizer, qa_decoder_choices_input_strs,
                                                                         input_padding_id,
                                                                         args.max_seq_len, add_eos=False,
                                                                         make_mask=True)
    qa_decoder_choices_labels = make_t5_tensor(tokenizer, qa_decoder_choices_label_strs, label_padding_id,
                                               args.max_seq_len, add_eos=False, make_mask=False)
    if args.label_to_use == 't5':
        qa_choice_label_list = [int(example.input_dict['t5_prediction']) for example in examples]
    else:
        qa_choice_label_list = [example.statement_label for example in examples]
    qa_choice_labels = torch.tensor(qa_choice_label_list, dtype=torch.long)
    exp_encoder_inputs, exp_encoder_masks = make_t5_tensor(tokenizer, exp_encoder_input_strs,
                                                           input_padding_id, args.max_seq_len, add_eos=False,
                                                           make_mask=True)
    exp_decoder_inputs, exp_decoder_masks = make_t5_tensor(tokenizer, exp_decoder_input_strs,
                                                           input_padding_id, args.max_seq_len, add_eos=True,
                                                           make_mask=True)
    exp_decoder_labels = make_t5_tensor(tokenizer, exp_decoder_label_strs, label_padding_id, args.max_seq_len,
                                        add_eos=True, make_mask=False)
    exp_context_ids = make_t5_tensor(tokenizer, exp_context_strs, input_padding_id, args.max_seq_len,
                                     add_eos=False, make_mask=False)
    exp_explanation_ids = make_t5_tensor(tokenizer, exp_explanation_strs, input_padding_id, args.max_seq_len,
                                         add_eos=True, make_mask=False)

    return [qa_encoder_inputs, qa_encoder_masks,
            qa_decoder_answer_inputs, qa_decoder_answer_masks, qa_decoder_answer_labels,
            qa_decoder_choices_inputs, qa_decoder_choices_masks, qa_decoder_choices_labels,
            qa_choice_labels,
            exp_encoder_inputs, exp_encoder_masks,
            exp_decoder_inputs, exp_decoder_masks, exp_decoder_labels,
            exp_context_ids, exp_explanation_ids]


def make_t5_cqa_inputs(args, tokenizer, examples: Sequence[CQAExample]):
    qa_encoder_input_strs = []
    qa_decoder_answer_input_strs = []
    qa_decoder_answer_label_strs = []
    qa_decoder_choices_input_strs = []
    qa_decoder_choices_label_strs = []
    exp_encoder_input_strs = []
    exp_decoder_input_strs = []
    exp_decoder_label_strs = []
    exp_context_strs = []
    exp_explanation_strs = []

    qa_encoder_x_masks = []  # e masked as 0
    qa_encoder_e_masks = []  # x masked as 0

    for idx, example in enumerate(examples):
        question_str = f'{example.question}'
        choices_str = f'The choices are {example.choices[0]}, {example.choices[1]} and {example.choices[2]}'

        # truncate question str if necessary
        question_str = truncate_question_str(args, tokenizer, question_str, choices_str)

        if args.label_to_use == 't5':
            answer_str = example.choices[int(example.input_dict['t5_prediction'])]
        else:
            answer_str = example.choices[example.label] if example.label >= 0 else ''

        if args.explanation_to_use == 't5':
            explanation_str = example.input_dict['t5_explanation']
        else:
            explanation_str = example.human_explanation

        if args.explanation_only:
            qa_encoder_input_str = f'task: [CLS] {choices_str} [SEP] My commonsense tells me {explanation_str}'
        elif args.condition_on_explanation:
            qa_encoder_input_str = f'task: [CLS] {question_str} {choices_str} [SEP] My commonsense tells me {explanation_str}'
        else:
            qa_encoder_input_str = f'task: [CLS] {question_str} {choices_str} [SEP]'
        exp_encoder_input_str = f'explain: [CLS] {question_str} {choices_str} [SEP]'

        # x,e masks
        x_len = len(tokenizer.encode(f'task: [CLS] {question_str} {choices_str} [SEP] '))
        qa_encoder_x_mask = [1] * x_len + [0] * (args.max_seq_len - x_len)
        qa_encoder_x_masks.append(qa_encoder_x_mask)
        start_len = len(tokenizer.encode('task: [CLS] '))
        que_len = len(tokenizer.encode(f'task: [CLS] {question_str} '))
        qa_encoder_e_mask = [1] * start_len + [0] * (que_len - start_len) + [1] * (args.max_seq_len - que_len)
        qa_encoder_e_masks.append(qa_encoder_e_mask)

        qa_decoder_answer_input_str = f'The answer is: {answer_str}'
        qa_decoder_answer_label_str = qa_decoder_answer_input_str
        qa_decoder_choices_input_str = [f'The answer is: {choice}' for choice in example.choices]
        qa_decoder_choices_label_str = qa_decoder_choices_input_str

        exp_decoder_input_str = f'My commonsense tells me {explanation_str}'
        exp_decoder_label_str = exp_decoder_input_str
        if args.rationalize:
            exp_context_str = [f'The answer is {choice} because ' for choice in example.choices]
        else:
            exp_context_str = ['My commonsense tells me ' for choice in example.choices]
        exp_explanation_str = explanation_str

        qa_encoder_input_strs.append(qa_encoder_input_str)
        qa_decoder_answer_input_strs.append(qa_decoder_answer_input_str)
        qa_decoder_answer_label_strs.append(qa_decoder_answer_label_str)
        qa_decoder_choices_input_strs.append(qa_decoder_choices_input_str)
        qa_decoder_choices_label_strs.append(qa_decoder_choices_label_str)
        exp_encoder_input_strs.append(exp_encoder_input_str)
        exp_decoder_input_strs.append(exp_decoder_input_str)
        exp_decoder_label_strs.append(exp_decoder_label_str)
        exp_context_strs.append(exp_context_str)
        exp_explanation_strs.append(exp_explanation_str)

    qa_encoder_x_masks = torch.tensor(qa_encoder_x_masks, dtype=torch.long)
    qa_encoder_e_masks = torch.tensor(qa_encoder_e_masks, dtype=torch.long)

    input_padding_id = tokenizer.pad_token_id
    label_padding_id = -100
    qa_encoder_inputs, qa_encoder_masks = make_t5_tensor(tokenizer, qa_encoder_input_strs, input_padding_id,
                                                         args.max_seq_len, add_eos=False, make_mask=True)
    qa_decoder_answer_inputs, qa_decoder_answer_masks = make_t5_tensor(tokenizer, qa_decoder_answer_input_strs,
                                                                       input_padding_id,
                                                                       args.max_seq_len, add_eos=False,
                                                                       make_mask=True)
    qa_decoder_answer_labels = make_t5_tensor(tokenizer, qa_decoder_answer_label_strs, label_padding_id,
                                              args.max_seq_len, add_eos=False, make_mask=False)
    qa_decoder_choices_inputs, qa_decoder_choices_masks = make_t5_tensor(tokenizer, qa_decoder_choices_input_strs,
                                                                         input_padding_id,
                                                                         args.max_seq_len, add_eos=False,
                                                                         make_mask=True)
    qa_decoder_choices_labels = make_t5_tensor(tokenizer, qa_decoder_choices_label_strs, label_padding_id,
                                               args.max_seq_len, add_eos=False, make_mask=False)
    if args.label_to_use == 't5':
        qa_choice_label_list = [int(example.input_dict['t5_prediction']) for example in examples]
    else:
        qa_choice_label_list = [example.label for example in examples]
    qa_choice_labels = torch.tensor(qa_choice_label_list, dtype=torch.long)
    exp_encoder_inputs, exp_encoder_masks = make_t5_tensor(tokenizer, exp_encoder_input_strs,
                                                           input_padding_id, args.max_seq_len, add_eos=False,
                                                           make_mask=True)
    exp_decoder_inputs, exp_decoder_masks = make_t5_tensor(tokenizer, exp_decoder_input_strs,
                                                           input_padding_id, args.max_seq_len, add_eos=True,
                                                           make_mask=True)
    exp_decoder_labels = make_t5_tensor(tokenizer, exp_decoder_label_strs, label_padding_id, args.max_seq_len,
                                        add_eos=True, make_mask=False)
    exp_context_ids = make_t5_tensor(tokenizer, exp_context_strs, input_padding_id, args.max_seq_len,
                                     add_eos=False, make_mask=False)
    exp_explanation_ids = make_t5_tensor(tokenizer, exp_explanation_strs, input_padding_id, args.max_seq_len,
                                         add_eos=True, make_mask=False)

    return [qa_encoder_inputs, qa_encoder_masks, qa_encoder_x_masks, qa_encoder_e_masks,
            qa_decoder_answer_inputs, qa_decoder_answer_masks, qa_decoder_answer_labels,
            qa_decoder_choices_inputs, qa_decoder_choices_masks, qa_decoder_choices_labels,
            qa_choice_labels,
            exp_encoder_inputs, exp_encoder_masks,
            exp_decoder_inputs, exp_decoder_masks, exp_decoder_labels,
            exp_context_ids, exp_explanation_ids]


def make_t5_nli_inputs(args, tokenizer, examples: Sequence[NLIExample]):
    qa_encoder_input_strs = []
    qa_decoder_answer_input_strs = []
    qa_decoder_answer_label_strs = []
    qa_decoder_choices_input_strs = []
    qa_decoder_choices_label_strs = []
    exp_encoder_input_strs = []
    exp_decoder_input_strs = []
    exp_decoder_label_strs = []
    exp_context_strs = []
    exp_explanation_strs = []

    for idx, example in enumerate(examples):
        premise_str = example.premise
        hypothesis_str = example.hypothesis

        if args.label_to_use == 't5':
            answer_str = example.choices[int(example.input_dict['t5_prediction'])]
        else:
            answer_str = example.choices[int(example.label)]

        if args.explanation_to_use == 't5':
            explanation_str = example.input_dict['t5_explanation']
        else:
            explanation_str = example.human_explanation

        qa_encoder_input_str = f'task: nli premise: [CLS] {premise_str} [SEP] hypothesis: {hypothesis_str} [SEP]'
        if args.condition_on_explanation:
            qa_encoder_input_str = f'{qa_encoder_input_str} My commonsense tells me {explanation_str}'
        exp_encoder_input_str = f'explain: nli premise: [CLS] {premise_str} [SEP] hypothesis: {hypothesis_str} [SEP]'

        qa_decoder_answer_input_str = f'answer {answer_str}'
        qa_decoder_answer_label_str = qa_decoder_answer_input_str
        qa_decoder_choices_input_str = [f'answer {choice}' for choice in example.choices]
        qa_decoder_choices_label_str = qa_decoder_choices_input_str

        exp_decoder_input_str = f'My commonsense tells me {explanation_str}'
        exp_decoder_label_str = exp_decoder_input_str
        exp_context_str = ['My commonsense tells me ' for choice in example.choices]
        exp_explanation_str = explanation_str

        qa_encoder_input_strs.append(qa_encoder_input_str)
        qa_decoder_answer_input_strs.append(qa_decoder_answer_input_str)
        qa_decoder_answer_label_strs.append(qa_decoder_answer_label_str)
        qa_decoder_choices_input_strs.append(qa_decoder_choices_input_str)
        qa_decoder_choices_label_strs.append(qa_decoder_choices_label_str)
        exp_encoder_input_strs.append(exp_encoder_input_str)
        exp_decoder_input_strs.append(exp_decoder_input_str)
        exp_decoder_label_strs.append(exp_decoder_label_str)
        exp_context_strs.append(exp_context_str)
        exp_explanation_strs.append(exp_explanation_str)

    input_padding_id = tokenizer.pad_token_id
    label_padding_id = -100
    qa_encoder_inputs, qa_encoder_masks = make_t5_tensor(tokenizer, qa_encoder_input_strs, input_padding_id,
                                                         args.max_seq_len, add_eos=False, make_mask=True)
    qa_decoder_answer_inputs, qa_decoder_answer_masks = make_t5_tensor(tokenizer, qa_decoder_answer_input_strs,
                                                                       input_padding_id,
                                                                       args.max_seq_len, add_eos=False,
                                                                       make_mask=True)
    qa_decoder_answer_labels = make_t5_tensor(tokenizer, qa_decoder_answer_label_strs, label_padding_id,
                                              args.max_seq_len, add_eos=False, make_mask=False)
    qa_decoder_choices_inputs, qa_decoder_choices_masks = make_t5_tensor(tokenizer, qa_decoder_choices_input_strs,
                                                                         input_padding_id,
                                                                         args.max_seq_len, add_eos=False,
                                                                         make_mask=True)
    qa_decoder_choices_labels = make_t5_tensor(tokenizer, qa_decoder_choices_label_strs, label_padding_id,
                                               args.max_seq_len, add_eos=False, make_mask=False)
    if args.label_to_use == 't5':
        qa_choice_label_list = [int(example.input_dict['t5_prediction']) for example in examples]
    else:
        qa_choice_label_list = [example.label for example in examples]
    qa_choice_labels = torch.tensor(qa_choice_label_list, dtype=torch.long)
    exp_encoder_inputs, exp_encoder_masks = make_t5_tensor(tokenizer, exp_encoder_input_strs,
                                                           input_padding_id, args.max_seq_len, add_eos=False,
                                                           make_mask=True)
    exp_decoder_inputs, exp_decoder_masks = make_t5_tensor(tokenizer, exp_decoder_input_strs,
                                                           input_padding_id, args.max_seq_len, add_eos=True,
                                                           make_mask=True)
    exp_decoder_labels = make_t5_tensor(tokenizer, exp_decoder_label_strs, label_padding_id, args.max_seq_len,
                                        add_eos=True, make_mask=False)
    exp_context_ids = make_t5_tensor(tokenizer, exp_context_strs, input_padding_id, args.max_seq_len,
                                     add_eos=False, make_mask=False)
    exp_explanation_ids = make_t5_tensor(tokenizer, exp_explanation_strs, input_padding_id, args.max_seq_len,
                                         add_eos=True, make_mask=False)

    return [qa_encoder_inputs, qa_encoder_masks,
            qa_decoder_answer_inputs, qa_decoder_answer_masks, qa_decoder_answer_labels,
            qa_decoder_choices_inputs, qa_decoder_choices_masks, qa_decoder_choices_labels,
            qa_choice_labels,
            exp_encoder_inputs, exp_encoder_masks,
            exp_decoder_inputs, exp_decoder_masks, exp_decoder_labels,
            exp_context_ids, exp_explanation_ids]


def make_t5_tensor(tokenizer, input_strs, pad_token_id, max_seq_len, add_eos: bool, make_mask: bool):
    all_input_ids = []
    for input_str in input_strs:
        if isinstance(input_str, str):
            input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
            input_ids += [tokenizer.eos_token_id] if add_eos else []
            truncate_seq_pair(input_ids, [], max_seq_len)
            input_ids += [pad_token_id] * (max_seq_len - len(input_ids))  # padding
            all_input_ids.append(input_ids)
        else:
            input_ids = []
            for choice_str in input_str:
                choice_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(choice_str))
                choice_ids += [tokenizer.eos_token_id] if add_eos else []
                truncate_seq_pair(choice_ids, [], max_seq_len)
                choice_ids += [pad_token_id] * (max_seq_len - len(choice_ids))  # padding
                input_ids.append(choice_ids)
            all_input_ids.append(input_ids)

    tensor = torch.tensor(all_input_ids, dtype=torch.long)
    if make_mask:
        mask = (tensor != pad_token_id).float()
        return tensor, mask
    else:
        return tensor


def truncate_question_str(args, tokenizer, question_str, choices_str):
    initial_len = len(tokenizer.encode(f'[CLS] {question_str} {choices_str} [SEP]'))
    exp_len = len(tokenizer.encode('My commonsense tells me ')) + args.max_sample_len
    prefix_len = len(tokenizer.encode('task: '))
    cap_len = args.max_seq_len - exp_len - prefix_len
    if initial_len > cap_len:
        over_by = initial_len - cap_len
        question_tokens = tokenizer.encode(question_str)
        keep_up_to = len(question_tokens) - over_by - 1
        new_question_tokens = question_tokens[:keep_up_to]
        question_str = tokenizer.decode(new_question_tokens) + '?'
    return question_str


def print_t5_input(args, tokenizer, input: T5Input, msg='T5Input'):
    ignore_tokens_list = [tokenizer.pad_token, '[UNK]']
    encoder_input_strs = detok_batch(tokenizer, input.encoder_inputs, ignore_tokens=ignore_tokens_list,
                                     eos_token=tokenizer.eos_token)
    decoder_input_strs = detok_batch(tokenizer, input.decoder_inputs, ignore_tokens=ignore_tokens_list,
                                     eos_token=tokenizer.eos_token)
    decoder_label_strs = detok_batch(tokenizer, input.decoder_labels, ignore_tokens=ignore_tokens_list,
                                     eos_token=tokenizer.eos_token)
    print(f'\n----{msg}----\n')
    print(f'encoder_input_strs: {encoder_input_strs}')
    print(f'encoder_inputs[0]: {input.encoder_inputs[0]}')
    print(f'encoder_masks[0]: {input.encoder_masks[0]}')
    print(f'decoder_input_strs: {decoder_input_strs}')
    print(f'decoder_label_strs: {decoder_label_strs}')
    if args.verbose:
        print(f'decoder_inputs[0]: {input.decoder_inputs[0]}')
        print(f'decoder_masks[0]: {input.decoder_masks[0]}')
        print(f'decoder_labels[0]: {input.decoder_labels[0]}')
    if input.choice_labels is not None:
        print(f'choice_labels: {input.choice_labels}')
    if input.context_ids is not None:
        context_strs = detok_batch(tokenizer, input.context_ids, ignore_tokens=ignore_tokens_list,
                                   eos_token=tokenizer.eos_token)
        if args.verbose:
            print(f'context_ids[0]: {input.context_ids[0]}')
        print(f'context_strs: {context_strs}')
    if input.explanation_ids is not None:
        explanation_strs = detok_batch(tokenizer, input.explanation_ids, ignore_tokens=ignore_tokens_list,
                                       eos_token=tokenizer.eos_token)
        if args.verbose:
            print(f'explanation_ids[0]: {input.explanation_ids[0]}')
        print(f'explanation_strs: {explanation_strs}')
    print('')


def print_t5_output(args, tokenizer, output: T5Output, msg='T5Output'):
    ignore_tokens_list = [tokenizer.pad_token]
    print(f'\n----{msg}----\n')
    print(f'encoder_hidden_states.size(): {output.encoder_hidden_states.size()}')
    if args.verbose:
        print(f'encoder_hidden_states: {output.encoder_hidden_states}')
    print(f'loss.size(): {output.loss.size()}')
    print(f'loss: {output.loss}')
    if output.choices_loss is not None:
        print(f'choices_loss: {output.choices_loss}')
    if output.predictions is not None:  # predictions can be either (batch_size, 1) or (batch_size, max_seq_len)
        if isinstance(output.predictions[0], list):
            prediction_strs = detok_batch(tokenizer, output.predictions, ignore_tokens=ignore_tokens_list,
                                          eos_token=tokenizer.eos_token)
            if args.verbose:
                print(f'prediction_ids[0]: {output.predictions[0]}')
            print(f'prediction_strs: {prediction_strs}')
        else:
            print(f'predictions: {output.predictions}')
    if output.acc_sum is not None:
        print(f'accuracy_sum: {output.acc_sum}')
    if output.bleu is not None:
        print(f'bleu: {output.bleu}')
    print('')


def sample_batched(model, context_ids, tokenizer, max_sample_len, model_name='T5',
                   input_ids=None, input_masks=None, encoder_hidden_states=None,
                   sampling_strategy='argmax', pad_prefix=True):
    '''
    Uses model to sample based on context_ids, until max_sample_len is hit, with the expectation that decoding will stop at a specified [end] token

    This function is batched, meaning predictions are placed at the end of each running sequence within a tensor of shape (batch_size x num_choices x max_seq_len)
    Before returning samples, the original contexts in running_contexts are set to the pad_token_id
    '''

    batch_size = context_ids.size(0)
    num_choices = context_ids.size(1)
    vocab_size = len(tokenizer)  # NOT tokenizer.vocab_size, this attr does not update when tokens are added
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    running_contexts = context_ids.clone()
    device = context_ids.device

    if model_name == 'T5':
        if encoder_hidden_states is None:
            encoder_outputs = model(input_ids=input_ids,
                                    attention_mask=input_masks)
            encoder_hidden_states = encoder_outputs[1]

        if input_masks.shape != context_ids.shape:
            input_masks = input_masks.unsqueeze(1).expand_as(context_ids)
            expand_shape = list(encoder_hidden_states.shape)
            expand_shape.insert(1, context_ids.size(1))
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1).expand(expand_shape)

        # flatten for T5.forward
        batch_size_by_num_choices = list(encoder_hidden_states.shape[:2])
        seq_len = encoder_hidden_states.size(2)
        embed_dim = encoder_hidden_states.size(3)

        encoder_hidden_states = encoder_hidden_states.reshape(-1, seq_len, embed_dim)
        input_masks = input_masks.reshape(-1, seq_len)

    # BEGIN SAMPLING
    for k in range(max_sample_len):

        attention_mask = (running_contexts != pad_token_id).float()

        # get locations of last non-pad tokens in each sequence for purposes of: getting predictions from logits, and updating running_contexts
        # print(running_contexts)
        where_last_tokens = [[question[choice_id].index(pad_token_id) - 1 for choice_id in range(num_choices)] for
                             question in running_contexts.tolist()]
        mask = torch.zeros(batch_size, num_choices, context_ids.size(2), vocab_size)
        mask = mask.to(device).float()
        for i in range(running_contexts.size(0)):
            for j in range(num_choices):
                last_token_index = where_last_tokens[i][j]
                mask[i, j, last_token_index, :] = 1

        # hold onto the starting point of sampling for each context
        if k == 0: init_where_last_tokens = where_last_tokens

        with torch.no_grad():
            if 'gpt' in model_name:
                outputs = model(running_contexts, attention_mask=attention_mask)
            elif 'T5' == model_name:
                running_contexts = running_contexts.view(-1, seq_len)
                attention_mask = attention_mask.view(-1, seq_len)
                outputs = model(encoder_hidden_states=encoder_hidden_states,
                                encoder_attention_mask=input_masks,
                                decoder_input_ids=running_contexts,
                                decoder_attention_mask=attention_mask)

            logits = outputs[0]

            # unflatten for T5
            if 'T5' == model_name:
                running_contexts = running_contexts.view(batch_size, num_choices, seq_len)
                logits = logits.view(batch_size, num_choices, seq_len, vocab_size)

            # get logits corresponding to last tokens in each sequence
            logits = logits * mask
            logits = torch.sum(logits, dim=2)  # (batch_size, num_choices, vocab_size)

            if sampling_strategy == 'argmax':
                preds = torch.argmax(logits, dim=-1)
            else:
                probs = torch.nn.functional.softmax(logits.squeeze(1), dim=1)  # (batch_size, vocab_size)
                preds = torch.multinomial(probs, num_samples=1)

        # assign preds to the first pad location in each running_contexts[i,j,:] sequence
        for i in range(batch_size):
            for j in range(num_choices):
                last_token_index = where_last_tokens[i][j]
                running_contexts[i, j, last_token_index + 1] = preds[i, j].item()

    samples = running_contexts
    if pad_prefix:
        for i in range(batch_size):
            for j in range(num_choices):
                end_of_context_index = init_where_last_tokens[i][j]
                samples[i, j, :(end_of_context_index + 1)] = pad_token_id

    return samples


def sample(device, model, prompts, encoder_hidden_states, input_masks, max_seq_length, tokenizer, decoder_masks=None,
           sampling_strategy='argmax'):
    if decoder_masks is None:
        decoder_masks = (prompts!=tokenizer.pad_token_id).int()
    context_lens = decoder_masks.sum(dim=-1)
    batch_size, num_choices, seq_len = list(decoder_masks.shape)
    finished = torch.zeros(batch_size, num_choices, dtype=torch.int32).to(device)
    vocab_size = len(tokenizer)
    while finished.sum().item() != batch_size*num_choices and decoder_masks.sum().item() != batch_size * num_choices * max_seq_length:
        prompts = prompts.view(-1, seq_len)
        input_masks = input_masks.view(-1, seq_len)
        with torch.no_grad():
            outputs = model(encoder_hidden_states = encoder_hidden_states,
                            encoder_attention_mask = input_masks,
                            decoder_input_ids = prompts,
                            decoder_attention_mask = decoder_masks)
        logits = outputs[0]
        prompts = prompts.view(batch_size, num_choices, seq_len)
        logits = logits.view(batch_size, num_choices, seq_len, vocab_size)
        if sampling_strategy == 'argmax':
            pred = torch.argmax(logits, dim=-1)
        elif sampling_strategy == 'multinomial':
            prob = torch.nn.functional.softmax(logits, dim=-1).view(-1, vocab_size)
            pred = torch.multinomial(prob, num_samples=1).view(batch_size, num_choices, seq_len)
        pred = torch.cat((torch.zeros((batch_size, num_choices, 1), dtype=torch.long).to(device), pred[..., :-1]), dim=2)
        prompts = decoder_masks * prompts + (1 - decoder_masks) * pred
        new_masks = torch.cat((torch.ones((batch_size, num_choices, 1), dtype=torch.int32).to(device), decoder_masks[..., :-1]), dim=2)
        new_tokens = (1 - decoder_masks) * new_masks * prompts
        finished += (torch.ones(batch_size, num_choices, dtype=torch.int32).to(device) - finished) * \
                    (new_tokens.sum(dim=2) == tokenizer.eos_token_id).int()
        decoder_masks = new_masks
    return prompts