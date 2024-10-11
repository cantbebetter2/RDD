import torch
import os
import copy
from collections import OrderedDict

base_ckpt = './logs/CIFAR10/8'
ckpt_teacher = torch.load(os.path.join(base_ckpt, 'ckpt.pt'), map_location='cpu')

time_scale = ckpt_teacher['time_scale']

if time_scale == 1:
    ckpt_teacher = ckpt_teacher['ema_model']
else:
    ckpt_teacher = ckpt_teacher['net_model']


print(type(ckpt_teacher))
print(ckpt_teacher.keys())

od=OrderedDict()

down_map = {
    '0':'0',
    '2':'1',
    '3':'2',
    '5':'3',
    '6':'4',
    '8':'5',
    '9':'6',
}

up_map = {
    '0':'0',
    '2':'1',
    '3':'2',
    '4':'3',
    '6':'4',
    '7':'5',
    '8':'6',
    '10':'7',
    '11':'8',
    '12':'9',
    '14':'10',
}

for k, v in ckpt_teacher.items():
    if 'downblocks' in k:
        idx = k.split('.')[1]
        if idx in down_map.keys():
            new_k = k.replace(idx, down_map[idx], 1)
            od[new_k] = v
    elif 'upblocks' in k:
        idx = k.split('.')[1]
        if idx in up_map.keys():
            new_k = k.replace(idx, up_map[idx], 1)
            od[new_k] = v
    else:
        od[k] = v

print(od.keys())

ckpt_teacher = torch.load(os.path.join(base_ckpt, 'ckpt.pt'), map_location='cpu')

new_pt = OrderedDict()

for k, v in ckpt_teacher.items():
    if time_scale == 1:
        if k == 'ema_model':
            new_pt[k] = od
        else:
            new_pt[k] = v
    else:
        if k == 'net_model':
            new_pt[k] = od
        else:
            new_pt[k] = v
    

torch.save(new_pt, './logs/ours/8_small/ckpt.pt')