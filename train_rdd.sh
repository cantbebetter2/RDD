python -m torch.distributed.launch --nproc_per_node=4 train_rdd.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 4,5,6,7 --num_gpus 4 \
    --logdir ./logs/8to4/rdd --base_ckpt ./logs/CIFAR10/8 \
    --classifier densenet201 --classifier_path ./classifier/result/cifar10/densenet201 \
    --num_workers 8 --feature --total_steps 20000 \
    --sample_step 5000 --save_step 5000 \
    --lr 5e-5 --wd 0. --loss_type mp2p --temperature 0.9

python ddim_eval.py --flagfile ./config/CIFAR10_EVAL.txt --logdir ./logs/8to4/rdd --gpu_id 4