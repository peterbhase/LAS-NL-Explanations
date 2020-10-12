#! /bin/sh
for seed in 11
do
  # generated rl explanations and predictions
  python t5_rl.py \
  --gpu=0 \
  --seed=${seed} \
  --model_name="cqa_t5_rl_ra_seed${seed}" \
  --train_task=True --explain_task=True --sample_task=True --train_sim=True \
  --do_rl=True --rl_sampling_strategy="multinomial" --ce_loss=True \
  --rationalize=True \
  --select_for="sim_acc" \
  --task_lr=5e-5 --sim_lr=1e-4 --alpha=0.95 --temperature=0.1 \
  --train_batch_size=2 --eval_batch_size=2 --grad_accumulation_factor=6 --num_train_epoch=20 \
  --dataset="cqa" --do_test=False \
  --max_seq_len=110 \
  --write_prediction \
  --train_output_file="cqa_ra_train_rl_seed${seed}.csv" \
  --eval_output_file="cqa_ra_eval_rl_seed${seed}.csv" \
  --test_output_file="cqa_ra_test_rl_seed${seed}.csv"

  # train task model using generated explanations and predictions
  python t5_rl.py \
  --gpu=0 \
  --seed=${seed} \
  --model_name="cqa_t5_ra_rl_sim_seed${seed}" \
  --train_data_file="cqa_ra_train_rl_seed${seed}.csv" \
  --eval_data_file="cqa_ra_eval_rl_seed${seed}.csv" \
  --test_data_file="cqa_ra_test_rl_seed${seed}.csv" \
  --task_base_model="t5-small" \
  --train_task=True --explain_task=False --sample_task=False \
  --train_sim=False --explain_sim=False \
  --do_rl=False \
  --select_for="task_acc" \
  --task_lr=1e-4 \
  --train_batch_size=12 --eval_batch_size=12 --grad_accumulation_factor=1 --num_train_epoch=10 \
  --condition_on_explanation=True --explanation_to_use="t5" --label_to_use="t5" \
  --dataset="cqa" --do_test=False \
  --max_seq_len=110
done
