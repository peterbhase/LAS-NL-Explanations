#! /bin/sh
for seed in 11
do
  # generated rl explanations and predictions
  python t5_rl.py \
  --seed=${seed} \
  --model_name="nli_t5_rl_ra_seed${seed}" \
  --train_task=True --explain_task=True --sample_task=True \
  --train_sim=True --explain_sim=False \
  --do_rl=True --rl_sampling_strategy="multinomial" --ce_loss=True \
  --select_for="sim_acc" \
  --task_lr=1e-5 --sim_lr=1e-4 --alpha=0.9 --temperature=0.1 \
  --train_batch_size=4 --eval_batch_size=4 --grad_accumulation_factor=3 --num_train_epoch=10 \
  --dataset="nli" --do_test=True \
  --train_data_file="train.tsv" \
  --eval_data_file="dev.tsv" \
  --test_data_file="test.tsv" \
  --max_seq_len=110 \
  --write_prediction \
  --train_output_file="train_rl_ra_seed${seed}.tsv" \
  --eval_output_file="eval_rl_ra_seed${seed}.tsv" \
  --test_output_file="test_rl_ra_seed${seed}.tsv"

  # train task model using generated explanations and predictions
  python t5_rl.py \
  --seed=${seed} \
  --model_name="nli_t5_rl_ra_sim_seed${seed}" \
  --task_base_model="t5-small" \
  --train_data_file="train_rl_ra_seed${seed}.tsv" \
  --eval_data_file="eval_rl_ra_seed${seed}.tsv" \
  --test_data_file="test_rl_ra_seed${seed}.tsv" \
  --train_task=True --explain_task=False --sample_task=False \
  --train_sim=False --explain_sim=False \
  --do_rl=False \
  --select_for="task_acc" \
  --task_lr=1e-4 \
  --train_batch_size=12 --eval_batch_size=12 --grad_accumulation_factor=1 --num_train_epoch=10 \
  --condition_on_explanation=True --explanation_to_use="t5" --label_to_use="t5" \
  --dataset="nli" --do_test=True \
  --max_seq_len=110
done
