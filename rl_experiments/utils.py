import argparse
import sys

import jsonlines
import csv
from sacrebleu import corpus_bleu


class SMExample(object):

    def __init__(self, input_dict):
        self.sm_id = input_dict['id']
        self.statements = [input_dict['sentence0'], input_dict['sentence1']]
        self.statement_label = 1 - int(input_dict['false'])
        self.explanations = [input_dict['A'], input_dict['B'], input_dict['C']]
        self.explanation_label = ['A', 'B', 'C'].index(input_dict['reason'])
        self.human_explanation = self.explanations[self.explanation_label]
        self.input_dict = input_dict


class CQAExample:

    def __init__(self, input_dict):
        self.cqa_id = input_dict['id']
        self.question = input_dict['question']
        self.choices = [input_dict['choice_0'], input_dict['choice_1'], input_dict['choice_2']]
        self.label = int(input_dict.get('label', -1))
        self.human_explanation = input_dict.get('human_expl_open-ended', '')
        self.input_dict = input_dict


class NLIExample(object):

    def __init__(self, input_dict):
        self.nli_id = input_dict['unique_key']
        self.premise = input_dict['premise']
        self.hypothesis = input_dict['hypothesis']
        self.human_explanation = input_dict['explanation1'] if 'explanation1' in input_dict else input_dict['explanation']
        self.choices = ['neutral', 'entailment', 'contradiction']
        self.label = int(input_dict['label'])
        self.input_dict = input_dict


class Report():
    """Report stores evaluation results during the training process as text files."""

    def __init__(self, args, file_path, score_names):
        self.fn = file_path
        self.args = args
        self.header = score_names
        self.text = ''

        # write input arguments at the top
        self.text += 'Input: python %s %s \n\n' % \
                     (sys.argv[0],
                      ' '.join([arg for arg in sys.argv[1:]]))

        # make header
        header_str = '%5s |' % 'epoch'
        for n, score_name in enumerate(self.header):
            header_str += ' %20s ' % score_name
            if n < len(score_names) - 1: header_str += '|'

        # write header
        self.blank_line = '-' * len(header_str)
        self.text += self.blank_line + \
                     f"\nTraining report for model: {args.model_name}" + \
                     '\n' + self.blank_line + \
                     '\n'
        self.text += header_str

    def write_epoch_scores(self, epoch, scores):
        # write scores
        self.text += '\n%5s |' % ('%d' % epoch)
        for idx, column_name in enumerate(self.header):
            self.text += ' %20s ' % ('%1.5f' % scores[column_name])
            if idx < len(scores) - 1: self.text += '|'
        self.__save()

    def write_final_score(self, args, final_score_str):
        self.text += '\n' + self.blank_line
        self.text += '\n%s' % final_score_str
        self.text += '\n' + self.blank_line + '\n'

        self.text += '\n'
        self.write_all_arguments(args)

        self.__save()

    def write_msg(self, msg):
        self.text += self.blank_line
        self.text += msg
        self.__save()

    def write_all_arguments(self, args):
        self.text += "\nAll arguments:\n"
        self.text += '\n'.join(['\t' + hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])
        self.__save()

    def full_print(self):
        print('\n' + self.text + '\n')

    def __save(self):
        if self.fn is not None:
            with open(self.fn, mode='w') as text_file:
                text_file.write(self.text)


def print_epoch_scores(epoch, scores):
    epoch_text = ' %5s |' % 'epoch'
    for n, score_name in enumerate(scores.keys()):
        epoch_text += ' %20s ' % score_name
        if n < len(scores) - 1: epoch_text += '|'
    epoch_text += '\n %5s |' % ('%d' % epoch)
    for n, score in enumerate(scores.values()):
        epoch_text += ' %20s ' % ('%1.5f' % score)
        if n < len(scores) - 1: epoch_text += '|'
    print(epoch_text + '\n')


def read_sm_examples(input_filepath):
    examples = []
    with jsonlines.open(input_filepath, 'r') as reader:
        for line in reader:
            examples.append(SMExample(line))
    return examples


def read_cqa_examples(input_filepath):
    examples = []
    with open(input_filepath, newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            examples.append(CQAExample(row))
    return examples


def read_nli_examples(input_filepath):
    examples = []
    with open(input_filepath, newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        for row in csv_reader:
            examples.append(NLIExample(row))
    return examples


def detok_batch(tokenizer, x, ignore_tokens=None, eos_token=None):
    '''
    - convert x into strings using tokenizer
    - x is either tensor of dim 2 or dim 3 or a .tolist() of such a tensor
    - stop decoding if eos_token hit, if eos_token provided
    - skip over tokens in ignore_tokens
    '''
    if ignore_tokens is not None:
        ignore_tokens_idx = tokenizer.convert_tokens_to_ids(ignore_tokens)
        ignore_tokens_idx += [-100, -1]
    else:
        ignore_tokens = []
        ignore_tokens_idx = [-100, -1]

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

            # check if any ignore_tokens are in decoded_sequence.
            # this is happening for some reason. many token_ids lead to [UNK], but [UNK] maps to id=100
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

                # check if any ignore_tokens are in decoded_sequence.
                # this is happening for some reason. many token_ids lead to [UNK], but [UNK] maps to id=100
                if any([ignore_token in decoded_sequence for ignore_token in ignore_tokens]):
                    decoded_sequence = ' '.join(
                        [token for token in decoded_sequence.split() if token not in ignore_tokens])

                # APPEND single decoding
                decoded_sequences.append(decoded_sequence)

            # APPEND list of n decodings
            texts.append(decoded_sequences)

    return texts


def str2bool(v):
    # used for boolean argparse values
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def write_prediction_to_sm_file(pred_dict, data_filepath, output_filepath):
    print(f'\nUpdating {data_filepath} with model predictions...')
    # read dataset
    examples = []
    with jsonlines.open(data_filepath, 'r') as reader:
        for line in reader:
            examples.append(line)

    # add or replace generated explanation to dictionary
    for column_name, predictions in pred_dict.items():
        if len(examples) != len(predictions):
            print('Warning: number of predictions not equal to number of input examples.')
            min_len = min(len(examples), len(predictions))
            examples = examples[:min_len]
            predictions = predictions[:min_len]
        for i, example in enumerate(examples):
            example[column_name] = predictions[i]

    with jsonlines.open(output_filepath, 'w') as writer:
        for line in examples:
            writer.write(line)
    print(f'Predictions written to {output_filepath} under columns {pred_dict.keys()}.')


def write_prediction_to_cqa_file(pred_dict, data_filepath, output_filepath):
    print(f'\nUpdating {output_filepath} with model predictions...')
    # read dataset
    examples = []
    with open(data_filepath, newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            examples.append(row)

    for column_name, predictions in pred_dict.items():
        if len(examples) != len(predictions):
            print('Warning: number of predictions not equal to number of input examples.')
            min_len = min(len(examples), len(predictions))
            examples = examples[:min_len]
            predictions = predictions[:min_len]
        for i, example in enumerate(examples):
            example[column_name] = predictions[i]

    # write to csv file
    with open(output_filepath, 'w', newline='') as csvfile:
        fieldnames = examples[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            writer.writerow(example)
    print(f'Predictions written to {output_filepath} under columns {pred_dict.keys()}.')


def write_prediction_to_nli_file(pred_dict, data_filepath, output_filepath):
    print(f'\nUpdating {output_filepath} with model predictions...')
    # read dataset
    examples = []
    with open(data_filepath, newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        for row in csv_reader:
            examples.append(row)

    for column_name, predictions in pred_dict.items():
        if len(examples) != len(predictions):
            print('Warning: number of predictions not equal to number of input examples.')
            min_len = min(len(examples), len(predictions))
            examples = examples[:min_len]
            predictions = predictions[:min_len]
        for i, example in enumerate(examples):
            example[column_name] = predictions[i]

    # write to csv file
    with open(output_filepath, 'w', newline='') as csvfile:
        fieldnames = examples[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for example in examples:
            writer.writerow(example)
    print(f'Predictions written to {output_filepath} under columns {pred_dict.keys()}.')


def compute_bleu(outputs, targets):
    # see https://github.com/mjpost/sacreBLEU
    targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
    return corpus_bleu(outputs, targets, lowercase=True).score
