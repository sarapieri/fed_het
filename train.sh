# Set CUDA_VISIBLE_DEVICES to only include GPU 6
export CUDA_VISIBLE_DEVICES=5

python train_FedAVG.py --FL_platform CAFormer-FedAVG \
    --dataset gldk23 --local_epochs 5 --max_communication_rounds 200 \
    --num_local_clients 20 --split_type real --seed 42 --n 3 --use_wandb 

