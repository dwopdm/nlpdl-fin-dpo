CUDA_VISIBLE_DEVICES=1,3,5 accelerate  launch --config_file ./ds_zero2.yaml ./dpo.py \
    --dataset_name /nfs-shared/data/hh-rlhf \
    --model_name_or_path /nfs-shared/models/pythia-2.8b \
    --learning_rate 5.0e-6 \
    --loss_type tpo \
    --beta 0.01 \
    --c1 0.01 \
    --c2 9 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir tpo \
    --no_remove_unused_columns