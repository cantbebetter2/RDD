#  Relational Diffusion Distillation for Efficient Image [PDF](https://arxiv.org/pdf/2410.07679)

This repository provides the implementation for our paper "Relational Diffusion Distillation for Efficient Image". Our approach introduces a novel relational distillation method for distilling fewer steps diffusion models, focusing on efficiency and performance.

## Environment

Python 3.8.18, torch 2.1.0

## Training

### Train the base model
```
python -m torch.distributed.launch --nproc_per_node=4 train_base.py \
    --flagfile ./config/CIFAR10_BASE.txt \
    --gpu_id 0,1,2,3 --logdir ./logs/CIFAR10/1024
```

### Distill using PD
```
python -m torch.distributed.launch --nproc_per_node=4 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 0,1,2,3 \
    --logdir ./logs/CIFAR10/512 --base_ckpt ./logs/CIFAR10/1024

...

python -m torch.distributed.launch --nproc_per_node=4 PD.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 0,1,2,3 \
    --logdir ./logs/CIFAR10/4 --base_ckpt ./logs/CIFAR10/8
```

### To use RDD, train the classifier using classifier/train.py first

```
python train.py --model densenet201
```

### Distill using RDD

```
python -m torch.distributed.launch --nproc_per_node=4 train_rdd.py \
    --flagfile ./config/CIFAR10_PD.txt --gpu_id 0,1,2,3 --num_gpus 4 \
    --logdir ./logs/8to4/rdd --base_ckpt ./logs/CIFAR10/8 \
    --classifier densenet201 --classifier_path ./classifier/result/cifar10/densenet201 \
    --num_workers 8 --feature --total_steps 20000 \
    --sample_step 5000 --save_step 5000 \
    --lr 5e-5 --wd 0. --loss_type mp2p --temperature 0.9
```



## Evaluation

### To eval, run score/get_npz.py first or download from [google drive](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing)

```
python get_npz.py --dataset cifar10
```

### Eval
```
# 8-step DDIM
python ddim_eval.py --flagfile ./config/CIFAR10_EVAL.txt --logdir ./logs/CIFAR10/1024 --stride 128
# 4-step PD
python ddim_eval.py --flagfile ./config/CIFAR10_EVAL.txt --logdir ./logs/CIFAR10/4
# 4-step RDD
python ddim_eval.py --flagfile ./config/CIFAR10_EVAL.txt --logdir ./logs/8to4/rdd
```
## Pre-trained Models
RCFD provide some pre-trained models (1024-step base model, 8-step PD-obtained model, and densenet201) in [google drive](https://drive.google.com/drive/folders/1iv_KPqjtDcHz4yOY8NQNCSTOMgdGu78N?usp=sharing). We use the same model as them.

## Citation
If you find this repository useful, please consider citing the following paper:
```
@misc{feng2024relationaldiffusiondistillationefficient,
      title={Relational Diffusion Distillation for Efficient Image Generation}, 
      author={Weilun Feng and Chuanguang Yang and Zhulin An and Libo Huang and Boyu Diao and Fei Wang and Yongjun Xu},
      year={2024},
      eprint={2410.07679},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.07679}, 
}
```

## Acknowledgment
This codebase is heavily borrowed from [RCFD](https://github.com/zju-SWJ/RCFD) , [pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm) and [diffusion_distiller](https://github.com/Hramchenko/diffusion_distiller).
