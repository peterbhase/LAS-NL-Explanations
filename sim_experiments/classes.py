import sys

class Report():
    """Report stores evaluation results during the training process as text files."""

    def __init__(self, args, file_path, score_names):
        self.fn = file_path
        self.args = args
        self.text = ''

        # write input arguments at the top
        self.text += 'Input: python %s %s \n\n' % \
                         (sys.argv[0], 
                          ' '.join([arg for arg in sys.argv[1:]]))

        # make header
        header = 'epoch |'
        for n, score_name in enumerate(score_names):
            header += ' %15s ' % score_name
            if n < len(score_names) - 1: header += '|'
        self.header = header

        # write header
        self.blank_line = '-'*len(header)
        self.text += self.blank_line + \
                    f"\nTraining report for model: {args.model_name}" + \
                    '\n' + self.blank_line + \
                    '\n'
        self.text += header


    def write_epoch_scores(self, epoch, scores):
        # write scores
        self.text += '\n%5s |' % str(epoch)
        for n, score in enumerate(scores.values()):
            self.text += ' %15s ' % ('%1.2f' % score)
            if n < len(scores) - 1: self.text += '|'
        self.__save()

    def write_final_score(self, args, final_score_str, time_msg=None):
        self.text += '\n' + self.blank_line
        self.text += '\n%s' % final_score_str
        self.text += '\n' + self.blank_line + '\n'

        if time_msg is not None:
            self.text += '\n%s\n' % time_msg
        
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

    def print_epoch_scores(self, epoch, scores, time_msg=None):
        epoch_text = ' epoch |'
        for n, score_name in enumerate(scores.keys()):
            epoch_text += ' %15s ' % score_name
            if n < len(scores) - 1: epoch_text += '|'
        epoch_text += '\n %5s |' % ('%d'% epoch)
        for n, score in enumerate(scores.values()):
            epoch_text += ' %15s ' % ('%1.2f' % score)
            if n < len(scores) - 1: epoch_text += '|'
        print(epoch_text + '\n')

    def full_print(self):
        print('\n' + self.text + '\n')

    def __save(self):
        if self.fn is not None:
            with open(self.fn, mode='w') as text_file:
                text_file.write(self.text)

