export WANDB_PROJECT=secalign-qwen3

python -m torch.distributed.run --nproc_per_node=4 --master_port=30015 align.py \
  --model_name_or_path mistralai/Mistral-7B-Instruct-v0.1 \
  --window_size 256 \
  --padding_side left \
  --data_path data/alpaca_data.json \
  --output_dir mistralai/Mistral-7B-Instruct-v0.1_dpo_NaiveCompletion_$(date +%Y-%m-%d-%H-%M-%S) \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 3 \
  --learning_rate 1.4e-4 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to wandb \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "MistralDecoderLayer" \
  --tf32 True \
  --attack NaiveCompletion \
  --lr_scale True \
  --downsample True \
  --alignment dpo \
  --model_max_length 512