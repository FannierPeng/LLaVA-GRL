## Contents
*[Install](#Install)
*[Dataset](#Dataset)
*[LLaVA-1.5 Checkpoints](#LLaVA-1.5 pre-trained checkpoints)
*[Inference](#Example of Inference Script)
*[Train](#Example of Training Script)
## Install
1. Clone this repository and navigate to LLaVA-GRL folder
```
git clone https://github.com/FannierPeng/LLaVA-GRL.git
cd LLaVA-GRL
```
2. Install Package
```
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```
3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
## Dataset
For data preparation, please refer to [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md)
## LLaVA-1.5 pre-trained checkpoints
Please refer to [LLaVA-1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/main)
## Example of Inference Script
```
#Merge Checkpoints
python scripts/merge_lora_weights.py --model-path "./checkpoints/llava-v1.5-7b-lora_imagenet_4shot_mask0d25-RL-routers_1"        --model-base "./checkpoints/llava-v1.5-7b"  --save-model-path "./checkpoints/llava-v1.5-7b-merged_imagenet_4shot_mask0d25-RL-routers_1"
#Inference
CUDA_VISIBLE_DEVICES=0 python playground/data/inference/imagenet.py
```
## Example of Training Script
```
WANDB_MODE=offline bash scripts/ft_lora_imagenet-4shot_mask0d25-RL-routers.sh
```
