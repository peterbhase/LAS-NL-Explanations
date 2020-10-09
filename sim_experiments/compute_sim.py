import pandas as pd
import numpy as np
import time
import os 
import argparse
from utils import str2bool


def run_analysis(args, gpu, data, model_name, explanations_to_use, labels_to_use, seed, split_name, model_size):
    '''
    compute sim metric for a model by writing to file (or checking if these in data)
    '''
    if data == 'QA':
        extension = 'csv'
        sep = ','
        folder = 'data/v1.0'
    elif data == 'NLI':
        extension = 'tsv'
        sep = '\t'        
        folder = 'data/e-SNLI-data'      
    save_dir = os.path.join(args.base_dir, 'saved_models')
    cache_dir = os.path.join(args.base_dir, 'cached_models')
    pretrained_name = args.task_pretrained_name + '-' + model_size
    train_file = os.path.join(folder, 'train.%s' % extension)
    dev_file = os.path.join(folder, 'dev.%s' % extension)
    test_file = os.path.join(folder, 'test.%s' % extension)
    write_base = 'preds' 
    xe_col = '%s_%s_%s_%s_seed%s_XE' % (write_base, data, pretrained_name, model_name, seed)
    e_col = '%s_%s_%s_%s_seed%s_E' % (write_base, data, pretrained_name, model_name, seed)
    x_col = '%s_%s_%s_%s_seed%s_X' % (write_base, data, pretrained_name, model_name, seed)

    train = pd.read_csv(train_file, sep=sep)
    dev = pd.read_csv(dev_file, sep=sep)
    test = pd.read_csv(test_file, sep=sep)
    to_use = dev if split_name == 'dev' else test
    script = 'main'
    if args.small_data:
        small_data_add = '-s -ss 100 '
    else:
        small_data_add = ''
    if xe_col not in to_use.columns or args.overwrite:
        print("\nWriting XE predictions...")
        os.system(f"python {script}.py --gpu {gpu} --model_name {model_name} --do_explain false --task_pretrained_name {pretrained_name} --multi_explanation false "
                  f"--data_dir {folder} --condition_on_explanations true --explanations_to_use {explanations_to_use} "
                  f"--dev_batch_size 20 "
                  f"--labels_to_use {labels_to_use} --do_train false --do_eval false --write_predictions --preds_suffix XE "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} --seed {seed} {small_data_add}"
          )
    if x_col not in to_use.columns or args.overwrite:
        print("Writing X predictions...")
        os.system(f"python {script}.py --gpu {gpu} --model_name {model_name} --do_explain false --task_pretrained_name {pretrained_name} --multi_explanation false "
                  f"--data_dir {folder} --condition_on_explanations false "
                  f"--dev_batch_size 20 "
                  f"--labels_to_use {labels_to_use} --do_train false --do_eval false --write_predictions --preds_suffix X "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} --seed {seed} {small_data_add}"
          )
    if e_col not in to_use.columns or args.overwrite:
        print("Writing E predictions...")
        os.system(f"python {script}.py --gpu {gpu} --model_name {model_name} --do_explain false --task_pretrained_name {pretrained_name} --multi_explanation false "
                  f"--data_dir {folder} --condition_on_explanations true --explanations_to_use {explanations_to_use} --explanations_only true "
                  f"--dev_batch_size 20 "
                  f"--labels_to_use {labels_to_use} --do_train false --do_eval false --write_predictions --preds_suffix E "
                  f"--save_dir {save_dir} --cache_dir {cache_dir} --seed {seed} {small_data_add}"
          )
    train = pd.read_csv(train_file, sep=sep)
    dev = pd.read_csv(dev_file, sep=sep)
    test = pd.read_csv(test_file, sep=sep)
    to_use = dev if split_name == 'dev' else test

    _ = compute_sim(args, to_use, labels_to_use, data, pretrained_name, model_name, seed, print_results = True)

    if args.bootstrap:
        start = time.time()
        boot_times = 10000
        print(f"Starting bootstrap with {boot_times/1000:.0f}k samples...")
        leaking_diff_list = []
        nonleaking_diff_list = []
        overall_metric_list = []
        for b in range(boot_times):
            boot_idx = np.random.choice(np.arange(len(to_use)), replace=True, size = len(to_use))    
            to_use_boot = to_use.iloc[boot_idx,:]  
            mean, leaking_diff, nonleaking_diff = compute_sim(args, to_use_boot, labels_to_use, data, pretrained_name, model_name, seed, print_results = False)
            overall_metric_list.append(mean)
            leaking_diff_list.append(leaking_diff)
            nonleaking_diff_list.append(nonleaking_diff)

        lb, ub = np.quantile(nonleaking_diff_list, (.025, .975))
        CI = (ub - lb) / 2
        print("\nnonleaking diff: %.2f (+/- %.2f)" % (np.mean(nonleaking_diff_list)*100, 100*CI))

        lb, ub = np.quantile(leaking_diff_list, (.025, .975))
        CI = (ub - lb) / 2
        print("\nleaking diff: %.2f (+/- %.2f)" % (np.mean(leaking_diff_list)*100, 100*CI))

        lb, ub = np.quantile(overall_metric_list, (.025, .975))
        CI = (ub - lb) / 2
        print("\nunweighted mean: %.2f (+/- %.2f)\n" % (np.mean(overall_metric_list)*100, 100*CI))

        print("time for bootstrap: %.1f minutes" % ((time.time() - start)/60))
        print("--------------------------\n")

 


def compute_sim(args, to_use, labels_to_use, data, pretrained_name, model_name, seed, print_results = False):
    labels = to_use[labels_to_use]
    xe_col = '%s_%s_%s_%s_seed%s_XE' % ('preds', data, pretrained_name, model_name, seed)
    e_col = '%s_%s_%s_%s_seed%s_E' % ('preds', data, pretrained_name, model_name, seed)
    x_col = '%s_%s_%s_%s_seed%s_X' % ('preds', data, pretrained_name, model_name, seed)
    xe = to_use[xe_col]
    e = to_use[e_col]
    x = to_use[x_col]    
    xe_correct = np.array(1*(labels==xe))
    x_correct = np.array(1*(labels==x))
    e_correct = np.array(1*(labels==e))

    # baseline and leaking proxy variable
    baseline_correct = 1*(x_correct)
    leaking = 1*(e_correct)
    leaked = np.argwhere(leaking.tolist()).reshape(-1)
    
    # get subgroups
    nonleaked = np.setdiff1d(np.arange(len(e_correct)), leaked)
    xe_correct_leaked = xe_correct[leaked]
    e_correct_leaked = e_correct[leaked]
    x_correct_leaked = x_correct[leaked]
    xe_correct_nonleaked = xe_correct[nonleaked]
    e_correct_nonleaked = e_correct[nonleaked]
    x_correct_nonleaked = x_correct[nonleaked]
    num_leaked = len(leaked)
    num_non_leaked = len(xe) - num_leaked

    unweighted_mean = np.mean([np.mean(xe_correct[split]) - np.mean(baseline_correct[split]) for split in [leaked,nonleaked]])
    nonleaking_diff = np.mean(xe_correct_nonleaked) - np.mean(baseline_correct[nonleaked])
    leaking_diff = np.mean(xe_correct_leaked) - np.mean(baseline_correct[leaked])
    if print_results:
        print("\n------------------------")
        print("num (probably) leaked: %d" % num_leaked)
        print("y|x,e : %.4f    baseline : %.4f     y|x,e=null: %.4f" % (np.mean(xe_correct_leaked), np.mean(baseline_correct[leaked]), np.mean(x_correct_leaked)))
        print("diff: %.4f" % (leaking_diff))
        print()
        print("num (probably) nonleaked: %d" % num_non_leaked)
        print("y|x,e : %.4f    baseline : %.4f     y|x,e=null: %.4f" % (np.mean(xe_correct_nonleaked), np.mean(baseline_correct[nonleaked]), np.mean(x_correct_nonleaked)))
        print("diff: %.4f" % (nonleaking_diff))
        print()
        print("overall: ")
        print("y|x : %.4f      y|e : %.4f" % (np.mean(x_correct), np.mean(e_correct)))
        print("y|x,e: %.4f     baseline : %.4f" % (np.mean(xe_correct), np.mean(baseline_correct)))
        print("\nunweighted mean: %.2f" % (unweighted_mean*100))
        print("--------------------------")
    return unweighted_mean, leaking_diff, nonleaking_diff

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument("--gpu", default=-1, type=int, help='')    
    parser.add_argument("--condition", default = "get_sim_metric", type=str, help='')    
    parser.add_argument("--data", default = 'NLI', help='')    
    parser.add_argument("--model_name", default ='', type=str, help='')   
    parser.add_argument("--explanations_to_use", default = 'ground_truth', type=str, help='')   
    parser.add_argument("--labels_to_use", default = 'label', type=str, help='')   
    parser.add_argument("--seed", default = '42', type=str, help='')  
    parser.add_argument('--leaking_param', default = 0, type=float, help='')
    parser.add_argument('--split_name', default='dev', type=str, help='see get_sim_metric')
    parser.add_argument('--task_pretrained_name', default='t5', type=str, help='')
    parser.add_argument('--model_size', default='base', type=str, help='')
    parser.add_argument('--server_number', '-s', default='13', type=str, help='')
    parser.add_argument('--bootstrap', action='store_true', help='')
    parser.add_argument('--small_data', action='store_true', help='Flag for using just a few datapoints for debugging purposes')
    parser.add_argument('--overwrite', action='store_true', help='rewrite predictions')
    parser.add_argument("--base_dir", default='', required=True, type=str, help="folders for saved_models and cached_models should be in this directory")
    args = parser.parse_args()

    if args.condition == "get_sim_metric":
        run_analysis(args,
                    args.gpu, 
                    args.data, 
                    args.model_name, 
                    args.explanations_to_use, 
                    args.labels_to_use, 
                    args.seed, 
                    args.split_name,
                    args.model_size)



