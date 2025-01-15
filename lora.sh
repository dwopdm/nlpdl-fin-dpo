#CUDA_VISIBLE_DEVICES=3,5 accelerate  launch --config_file ./deepspeed_zero2.yaml ./dpo.py \
CUDA_VISIBLE_DEVICES=7 python ./dpo.py \
    --dataset_name /nfs-shared/data/hh-rlhf/helpful-base \
    --model_name_or_path /nfs-shared/models/pythia-2.8b \
    --learning_rate 5.0e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir tpo2005 \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 16\
