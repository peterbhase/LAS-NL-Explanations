## Reproducing Experiments

Training task models and simulators is done with the `main.py` script, and running particular experiments can be done with `run_tasks.py` in the manner specified below. We give commands corresponding to the four graphical models we evaluate, simulators for each of these models, and the two-agent experiments with SGD optimization.

**Note** that for all modeling experiments below, `gpu`, `save_dir`, and `cache_dir` must be provided as args to argpase (recommended to make save_dir and cache_dir in same directory). `-b` and `-g` refer to train batch size and gradient accumulation factors, respectively (effective batch size is their product). 

Lastly, simply replace all instances of "QA" with "NLI" to run the analogous experiments for e-SNLI instead of CoS-E (and adjust the effective batch size to 36).

*Task Model*:  
`python run_tasks.py --gpu gpu -e QA.task -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir`

*Human Simulator*:  
`python run_tasks.py --gpu gpu -e QA.SIM.human -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir`

*MT-Re*:  
Task model: `python run_tasks.py --gpu gpu -e QA.CLM.reason.MT -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir`  
Simulator: `python run_tasks.py --gpu gpu -e QA.SIM.MT.RE -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir`  

*MT-Ra*:  
Task model: `python run_tasks.py --gpu gpu -e QA.CLM.rationalize.MT -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir`  
Simulator: `python run_tasks.py --gpu gpu -e QA.SIM.MT.RA -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir`  

*ST-Re*:  
Generator: `python run_tasks.py --gpu gpu -e QA.CLM.reason -b 6 -g 6 --save_dir save_dir --cache_dir cache_dir `  
Task model: `python run_tasks.py --gpu gpu -e QA.ST.RE -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir `  
Simulator: `python run_tasks.py --gpu gpu -e QA.SIM.ST.RE -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir`  

*ST-Ra*:  
Generator: `python run_tasks.py --gpu gpu -e QA.CLM.rationalize -b 6 -g 6 --save_dir save_dir --cache_dir cache_dir `  
Task model: `python run_tasks.py --gpu gpu -e QA.ST.RA -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir`  
Simulator: `python run_tasks.py --gpu gpu -e QA.SIM.ST.RA -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir `

Note that in our two-agent experiments we initialize the models with pretrained task models and simulators (using `--task_prefinetuned_name` and `sim_prefinetuned_name` args). 

*Two-agent CQA Reasoning*:  
`python T5-2-agent_main.py --gpu gpu --model_name pass.reason --task_pretrained_name t5-base --human_exp_coef .15 --task_coef .35 --suppress_coef .2 --X_coef .5 --train_batch_size 1 --grad_accumulation_factor 12 --multi_explanation false --save_dir save_dir --cache_dir cache_dir --data_dir data/v1.0`

*Two-agent CQA Rationalizing*:  
`python T5-2-agent_main.py --gpu gpu --model_name pass.rationalize --task_pretrained_name t5-base --human_exp_coef .15 --task_coef .35 --suppress_coef .2 --X_coef .5 --train_batch_size 1 --grad_accumulation_factor 12 --multi_explanation true --save_dir save_dir --cache_dir cache_dir --data_dir data/v1.0` 

*Two-agent NLI Reasoning*:  
`python T5-2-agent_main.py --gpu gpu --model_name pass.reason --task_pretrained_name t5-base --human_exp_coef .15 --task_coef .35 --suppress_coef .2  --X_coef .4 --E_coef .2 --train_batch_size 1 --grad_accumulation_factor 12 --multi_explanation false --save_dir save_dir --cache_dir cache_dir  --data_dir data/e-SNLI-data`

*Two-agent NLI Rationalizing*:  
`python T5-2-agent_main.py --gpu gpu --model_name pass.rationalize --task_pretrained_name t5-base --human_exp_coef .15 --task_coef .35 --suppress_coef .2  --X_coef .4 --E_coef .2 --train_batch_size 1 --grad_accumulation_factor 12 --multi_explanation true --save_dir save_dir --cache_dir cache_dir  --data_dir data/e-SNLI-data`


## Computing LAS

We compute LAS scores with the `compute_sim.py` script. Here, `gpu` and `base_dir` must be provided as arguments. `base_dir` should include a `saved_models` and `cached_models` directories. For each condition, LAS computation is possible after running the respective experiments from above. Note `split_name` is `dev` for CQA to compare with human provided explanations, but should be `test` for SNLI.

*Human Simulator*:  
`python compute_sim.py --model_name sim.human --explanations_to_use ground_truth --gpu gpu --split_name dev --data QA --seed seed --bootstrap`

*MT-Re*:  
`python compute_sim.py --model_name sim.MT.RE --explanations_to_use t5-MT-single-exp-seed21 --gpu gpu --split_name dev --data QA --seed 21 --bootstrap --labels_to_use preds_QA_t5-base_MT.RE_seed21`  

*MT-Ra*:  
`python compute_sim.py --model_name sim.MT.RA --explanations_to_use t5-MT-multi-exp-pred-seed21 --gpu gpu --split_name dev --data QA --seed 21 --bootstrap -s aws --overwrite --labels_to_use preds_QA_t5-base_MT.RA_seed21`  

`python compute_sim.py --model_name sim.ST.RE --explanations_to_use t5-single-exp-seed21 --gpu gpu --split_name dev --data QA --seed 21 --bootstrap --labels_to_use preds_QA_t5-base_ST.RE_seed21`  

*ST-Ra*:  
`python compute_sim.py --model_name sim.ST.RA --explanations_to_use t5-multi-exp-seed21 --gpu gpu --split_name dev --data QA --seed 21 --bootstrap --labels_to_use preds_QA_t5-base_ST.RA_seed21`  



