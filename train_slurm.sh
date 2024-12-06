#!/bin/bash
python scripts/train_text_to_image_sd.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2" \
    --variant "fp16" \
    --train_data_dir "/home/iqbal/diffusion_playground/lotus_dataset" \
    --output_dir "/netscratch/naeem/lotus_output/lotus_test" \
    --prediction_type "epsilon" \
    --resolution "512" \
    --train_batch_size "1" \
    --learning_rate "1e-5" \
    --max_train_steps "250" \
    --report_to "wandb" \
    --checkpointing_steps "100"
    # --resume_from_checkpoint "latest"