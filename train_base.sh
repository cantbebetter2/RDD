python -m torch.distributed.launch --nproc_per_node=4 train_base.py \
    --flagfile ./config/CIFAR10_BASE.txt \
    --gpu_id 0,1,2,3 --logdir ./logs/celeba_subset/1024 --num_gpus 4 \
    --dataset celeba 

# python ddim_eval.py --flagfile ./config/STL10_EVAL.txt --logdir ./logs/stl10/1024 --gpu_id 4