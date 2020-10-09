import os
import argparse

# --- BEGIN QA --- #

seed_variance_test = [21]

def QA_task(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --do_explain false "
                  f"--model_name baseline "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 20 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
        )

def QA_SIM_human(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use ground_truth "
                  f"--model_name sim.human --explanation_dropout .5 "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 30 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
        )

def QA_CLM_reason(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --do_task false --task_coef 0 --multi_explanation false --select_for bleu "
                  f"--model_name CLM.reason --write_predictions --max_sample_len 20 "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 5 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
        )

def QA_CLM_rationalize(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --do_task false --task_coef 0 --multi_explanation true --select_for bleu  "
                  f"--model_name CLM.rationalize --write_predictions --max_sample_len 20 "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 5 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
        ) 

def QA_CLM_reason_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --task_coef .5 --multi_explanation false "
                  f"--model_name MT.RE --write_predictions --max_sample_len 20 "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 20 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
        ) 

def QA_SIM_CLM_reason_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --task_pretrained_name t5-base --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-MT-single-exp-seed{seed} --labels_to_use preds_QA_t5-base_MT.RE_seed{seed} "
                  f"--model_name sim.MT.RE --explanation_dropout .5 "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 30 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
        ) 

def QA_CLM_rationalize_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --task_coef .5 --multi_explanation true "
                  f"--model_name MT.RA --write_predictions --max_sample_len 20 "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 20 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
        ) 

def QA_SIM_CLM_rationalize_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --task_pretrained_name t5-base --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-MT-multi-exp-pred-seed{seed} --labels_to_use preds_QA_t5-base_MT.RA_seed{seed} "
                  f"--model_name sim.MT.RA  --explanation_dropout .5 "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 30 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
        )

def QA_ST_reason(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-single-exp-seed{seed} "
                  f"--model_name ST.RE --write_predictions "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 20 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
        ) 

def QA_SIM_ST_reason(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --task_pretrained_name t5-base --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-single-exp-seed{seed} --labels_to_use preds_QA_t5-base_ST.RE_seed{seed} "
                  f"--model_name sim.ST.RE  --explanation_dropout .5 --print_examples "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 30 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
        ) 

def QA_ST_rationalize(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --do_explain false --multi_explanation true --condition_on_explanations true --explanations_to_use t5-multi-exp-seed{seed} "
                  f"--model_name ST.RA --write_predictions "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 20 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
        )

def QA_SIM_ST_rationalize(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --task_pretrained_name t5-base --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-multi-exp-pred-seed{seed} --labels_to_use preds_QA_t5-base_ST.RA_seed{seed} "
                  f"--model_name sim.ST.RA  --explanation_dropout .5 --print_examples "
                  f"--data_dir data/v1.0 --gpu {args.gpu} --seed {seed} --num_train_epochs 30 --warmup_proportion .1 "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} {small_data_addin} "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} "
        ) 

# --- BEGIN NLI --- #

def NLI_task(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --do_explain false "
                  f"--model_name baseline "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --num_train_epochs 10 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        )

def NLI_SIM_human(args):
    LR = 1e-4 if 't5' in args.model else 1e-5
    for seed in seed_variance_test:
        os.system(f"python main.py --do_explain false --task_pretrained_name {args.model} --multi_explanation false --condition_on_explanations true --explanations_to_use ground_truth "
                  f"--model_name sim.human --input_dropout .2 --explanation_dropout .4 --lr {LR} "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --num_train_epochs 15 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        )

def NLI_CLM_reason(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --do_task false --task_coef 0 --multi_explanation false --select_for bleu "
                  f"--model_name CLM.reason --write_predictions "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --num_train_epochs 5  "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        )

def NLI_CLM_rationalize(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --do_task false --task_coef 0 --multi_explanation true --select_for bleu " 
                  f"--model_name CLM.rationalize --write_predictions "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --num_train_epochs 5 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        ) 

def NLI_CLM_reason_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --task_coef .5 --multi_explanation false --do_train false --do_eval false " 
                  f"--model_name MT.RE --write_predictions "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor}  --warmup_proportion .01 --num_train_epochs 10 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        ) 

def NLI_SIM_CLM_reason_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --task_pretrained_name t5-base --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-MT-single-exp-seed{seed} --labels_to_use preds_NLI_t5-base_MT.RE_seed{seed} "
                  f"--model_name sim.MT.RE --input_dropout .2 --explanation_dropout .4  "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --num_train_epochs 15 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        ) 


def NLI_CLM_rationalize_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --task_coef .5 --multi_explanation true --do_train false --do_eval false " 
                  f"--model_name MT.RA --write_predictions "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor}  --warmup_proportion .01 --num_train_epochs 10 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        )

def NLI_SIM_CLM_rationalize_MT(args):
    for seed in seed_variance_test:
        os.system(f"python main.py --task_pretrained_name t5-base --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-MT-multi-exp-pred-seed{seed} --labels_to_use preds_NLI_t5-base_MT.RA_seed{seed} "
                  f"--model_name sim.MT.RA --input_dropout .2 --explanation_dropout .4 "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed}  "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --num_train_epochs 15 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        ) 

def NLI_ST_reason(args):
    LR = 1e-4 if 't5' in args.model else 1e-5
    for seed in seed_variance_test:
        os.system(f"python main.py --task_pretrained_name {args.model} --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-single-exp-seed{seed} "
                  f"--model_name ST.RE --write_predictions --lr {LR} "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor}  --warmup_proportion .01 --num_train_epochs 10 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        ) 

def NLI_SIM_ST_reason(args):
    LR = 1e-4 if 't5' in args.model else 1e-5
    for seed in seed_variance_test:
        os.system(f"python main.py --task_pretrained_name {args.model} --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-single-exp-seed{seed} --labels_to_use preds_NLI_{args.model}_ST.RE_seed{seed} "
                  f"--model_name sim.ST.RE  --input_dropout .2 --explanation_dropout .4 --lr {LR} "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed}  "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor}  --warmup_proportion .01 --num_train_epochs 15 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        ) 

def NLI_ST_rationalize(args):
    LR = 1e-4 if 't5' in args.model else 1e-5
    for seed in seed_variance_test:
        os.system(f"python main.py --task_pretrained_name {args.model} --do_explain false --multi_explanation true --condition_on_explanations true --explanations_to_use t5-multi-exp-seed{seed} "
                  f"--model_name ST.RA --write_predictions --lr {LR} "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --num_train_epochs 10 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        ) 

def NLI_SIM_ST_rationalize(args):
    LR = 1e-4 if 't5' in args.model else 1e-5
    for seed in seed_variance_test:
        os.system(f"python main.py --task_pretrained_name {args.model} --do_explain false --multi_explanation false --condition_on_explanations true --explanations_to_use t5-multi-exp-exp-seed{seed} --labels_to_use preds_NLI_{args.model}_ST.RA_seed{seed} "
                  f"--model_name sim.ST.RA  --input_dropout .2 --explanation_dropout .4 --lr {LR} "
                  f"--data_dir data/e-SNLI-data --gpu {args.gpu} --seed {seed} "
                  f"--train_batch_size {args.train_batch_size} --grad_accumulation_factor {args.grad_accumulation_factor} --warmup_proportion .01 --num_train_epochs 15 "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} {small_data_addin}"
        ) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0, type=int, help='')    
    parser.add_argument("--experiment", '-e', type=str, help='')
    parser.add_argument("--server_number", '-s', required=True, type=str, help='')
    parser.add_argument("--model", default='t5-base', type=str, help='HuggingFace transformer model')
    parser.add_argument("--train_batch_size", '-b', default=3, type=int, help="ONLY FOR QA. Total batch size for training. Effective batch size is this times grad_accumulation_factor")
    parser.add_argument('--grad_accumulation_factor', '-g', type=int, default=4, help="ONLY FOR QA. Number of updates steps to accumulate before performing a backward pass and step.")
    parser.add_argument('--small_data', action='store_true', help='Flag for using just a few datapoints for debugging purposes')
    parser.add_argument("--save_dir", default='', required=True, type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--cache_dir", default='', required=True, type=str,
                        help="Directory for cacheing pretrained models.")    
    args = parser.parse_args()
    save_dir = args.save_dir
    cache_dir = args.cache_dir

    if args.small_data:
        small_data_addin = '-s -ss 64 ' # uses 64 points per split in main.py
    else:
        small_data_addin = ''

    print("Starting experiment %s " % args.experiment)
    print("Using seeds ", seed_variance_test)
    print("Saving models in %s" % save_dir)

    # --- begin QA --- #

    if args.experiment == 'QA.task':
        QA_task(args) 

    if args.experiment == 'QA.SIM.human':
        QA_SIM_human(args) 

    if args.experiment == 'QA.CLM.reason':
        QA_CLM_reason(args) 

    if args.experiment == 'QA.CLM.rationalize':
        QA_CLM_rationalize(args) 

    if args.experiment == 'QA.CLM.reason.MT':
        QA_CLM_reason_MT(args) 

    if args.experiment == 'QA.SIM.MT.RE':
        QA_SIM_CLM_reason_MT(args) 

    if args.experiment == 'QA.CLM.rationalize.MT':
        QA_CLM_rationalize_MT(args) 

    if args.experiment == 'QA.SIM.MT.RA':
        QA_SIM_CLM_rationalize_MT(args) 

    if args.experiment == 'QA.ST.RE':
        QA_ST_reason(args) 

    if args.experiment == 'QA.SIM.ST.RE':
        QA_SIM_ST_reason(args) 

    if args.experiment == 'QA.ST.RA':
        QA_ST_rationalize(args) 

    if args.experiment == 'QA.SIM.ST.RA':
        QA_SIM_ST_rationalize(args)
   

    # --- begin NLI --- #

    if args.experiment == 'NLI.task':
        NLI_task(args) 

    if args.experiment == 'NLI.SIM.human':
        NLI_SIM_human(args) 

    if args.experiment == 'NLI.CLM.reason':
        NLI_CLM_reason(args) 

    if args.experiment == 'NLI.CLM.rationalize':
        NLI_CLM_rationalize(args) 

    if args.experiment == 'NLI.CLM.reason.MT':
        NLI_CLM_reason_MT(args) 

    if args.experiment == 'NLI.SIM.MT.RE':
        NLI_SIM_CLM_reason_MT(args) 

    if args.experiment == 'NLI.CLM.rationalize.MT':
        NLI_CLM_rationalize_MT(args) 

    if args.experiment == 'NLI.SIM.MT.RA':
        NLI_SIM_CLM_rationalize_MT(args) 

    if args.experiment == 'NLI.ST.RE':
        NLI_ST_reason(args) 

    if args.experiment == 'NLI.ST.RA':
        NLI_ST_rationalize(args) 

    if args.experiment == 'NLI.SIM.ST.RE':
        NLI_SIM_ST_reason(args) 

    if args.experiment == 'NLI.SIM.ST.RA':
        NLI_SIM_ST_rationalize(args) 
