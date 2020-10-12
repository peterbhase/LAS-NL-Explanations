## CQA Code

------

### Requirements

torch 1.4
transformers 2.5.1

### models

**models/T5ForMC.py** 
- defines *T5ModelForMC* wrapper for *T5PreTrainedModel*
- .forward computes the loss for an output sequence given an input sequence or encoder_hidden_states
- .QA_forward returns a loss of shape (batch_size x num_choices) given output sequence answers of shape (batch_size x num_choices x max_seq_len). predictions are the highest likelihood answer, and can be obtained by computing np.argmin(output_loss, axis = -1)

### experiment scripts

Below are the training scripts and experiment shell scripts. 

**t5_rl.py** - for experiments with multi-agent reinforcement learning using simulation metric as reward.

T5-RL-reason: run_cqa_t5_rl_re.sh, run_nli_t5_rl_re.sh

T5-RL-rationalize: run_cqa_t5_rl_ra.sh, run_nli_t5_rl_ra.sh