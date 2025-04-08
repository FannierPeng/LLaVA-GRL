## Examples
## Inference
CUDA_VISIBLE_DEVICES=0 python playground/data/inference/imagenet.py
## Merge checkpoints
python scripts/merge_lora_weights.py --model-path "./checkpoints/llava-v1.5-7b-lora_imagenet_4shot_mask0d25-RL-routers_1"        --model-base "./checkpoints/llava-v1.5-7b"  --save-model-path "./checkpoints/llava-v1.5-7b-merged_imagenet_4shot_mask0d25-RL-routers_1"
## Train
WANDB_MODE=offline bash scripts/ft_lora_imagenet-4shot_mask0d25-RL-routers.sh
