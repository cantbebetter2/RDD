python -m torch.distributed.launch --nproc_per_node=8 RCFD.py \
    --flagfile ./config/IMAGENET64_RCFD.txt --gpu_id 0,1,2,3,4,5,6,7 --num_gpus 8 --dataset imagenet64 \
    --logdir ./logs/imagenet/rcfd/4_temp0.75 --base_ckpt ./logs/imagenet/pd8  \
    --classifier densenet201 --classifier_path ./result \
    --temp 0.75 --alpha 0 --total_steps 50000 --sample_step 10000 --save_step 10000

python ddim_eval.py --flagfile ./config/IMAGENET64_EVAL.txt --logdir ./logs/imagenet/rcfd/4_temp0.75  --gpu_id 0