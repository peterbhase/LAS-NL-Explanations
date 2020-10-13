# Leakage-Adjusted Simulatability
This is the codebase for the paper:  
[Leakage-Adjusted Simulatability: Can Models Generate Non-Trivial Explanations of Their Behavior in Natural Language?](https://arxiv.org/abs/2010.04119)  
Peter Hase, Shiyue Zhang, Harry Xie, Mohit Bansal. Findings of EMNLP 2020

## Repository Structure

```
|__ sim_experiments/ --> Directory with models and experiment scripts
    |__ data/ --> includes eSNLI and CoS-E data used in experiments
    |__ models/ --> includes wrappers for T5 model for multiple choice tasks
    |__ training_reports/ --> includes wrappers for T5 model for multiple choice tasks
    |__ main.py --> train task model or simulator model on SNLI or CQA
    |__ T5-2-agent_main.py --> train models for multi-agent experiments [SGD] in paper
    |__ compute_sim.py --> script for computing LAS score for model explanations
    |__ run_tasks.py --> script for running experimental conditions across seeds
    |__ *.py --> utilities for data loading, explanation sampling, etc.
    |__ causal_estimation.Rmd --> R markdown script used to calibrate simulator models and compute LAS with *k* bins
|__ alternative_objectives/ --> Directory with additional experimental scripts
    |__ code coming soon
|__ rl_experiments/ --> Directory with code for multi-agent [RL] experimental scripts
    |__ see internal README
|__ human_experiments/ --> Directory with R code for analyzing human experiment data        
    |__ ratings_analysis.Rmd --> R markdown script used to analyze human quality ratings
    |__ expert_analysis.Rmd --> R markdown script used to analyze expert simulation data
    |__ more human experiment code coming soon
|__ requirements.txt

```

## Requirements

- Python 3.6 
- torch 1.4
- transformers 2.5.1
- sacrebleu
- pandas
- numpy

## Reproducing Experiments 

See READMEs in each directory for instructions on reproducing each set of experiments.
