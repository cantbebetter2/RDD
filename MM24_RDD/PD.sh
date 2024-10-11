python -m torch.distributed.launch --nproc_per_node=2 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 4,5 --num_gpus 2 \
    --logdir ./logs/celeba_subset/512 --base_ckpt ./logs/celeba_subset/1024 \
    --dataset celeba --num_workers 12

python -m torch.distributed.launch --nproc_per_node=2 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 4,5 --num_gpus 2 \
    --logdir ./logs/celeba_subset/256 --base_ckpt ./logs/celeba_subset/512 \
    --dataset celeba --num_workers 12

python -m torch.distributed.launch --nproc_per_node=2 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 4,5 --num_gpus 2 \
    --logdir ./logs/celeba_subset/128 --base_ckpt ./logs/celeba_subset/256 \
    --dataset celeba --num_workers 12

python -m torch.distributed.launch --nproc_per_node=2 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 4,5 --num_gpus 2 \
    --logdir ./logs/celeba_subset/64 --base_ckpt ./logs/celeba_subset/128 \
    --dataset celeba --num_workers 12

python -m torch.distributed.launch --nproc_per_node=2 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 4,5 --num_gpus 2 \
    --logdir ./logs/celeba_subset/32 --base_ckpt ./logs/celeba_subset/64 \
    --dataset celeba --num_workers 12

python -m torch.distributed.launch --nproc_per_node=2 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 4,5 --num_gpus 2 \
    --logdir ./logs/celeba_subset/16 --base_ckpt ./logs/celeba_subset/32 \
    --dataset celeba --num_workers 12

python -m torch.distributed.launch --nproc_per_node=2 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 4,5 --num_gpus 2 \
    --logdir ./logs/celeba_subset/8 --base_ckpt ./logs/celeba_subset/16 \
    --dataset celeba --num_workers 12


# python ddim_eval.py --flagfile ./config/IMAGENET64_EVAL.txt --logdir ./logs/imagenet64/2 --gpu_id 1