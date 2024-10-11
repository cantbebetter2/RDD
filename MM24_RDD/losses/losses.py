import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import einsum
from einops import rearrange
import numpy as np
from .diffkd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline
from .scheduling_ddim import DDIMScheduler

class IntraImageMiniBatch(nn.Module):
    def __init__(self, temperature=1.0):
        super(IntraImageMiniBatch, self).__init__()
        self.temperature = temperature
    
    def pair_wise_sim_map_speed(self, fea_0, fea_1):
        B, C, H, W = fea_0.size()

        fea_0 = fea_0.reshape(B, C, -1).transpose(1, 2)
        fea_1 = fea_1.reshape(B, C, -1)
        
        sim_map = torch.bmm(fea_0, fea_1)
        return sim_map.reshape(-1, sim_map.shape[-1])


    def forward(self, feat_S, feat_T):
        B, C, H, W = feat_S.size()
        
        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)
        
        sim_dis = torch.tensor(0.).cuda()
        s_sim_map = self.pair_wise_sim_map_speed(feat_S, feat_S)
        t_sim_map = self.pair_wise_sim_map_speed(feat_T, feat_T)

        p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
        p_t = F.softmax(t_sim_map / self.temperature, dim=1)

        sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
        sim_dis += sim_dis_
        
        return sim_dis

class Intra_sample_p2p_loss(nn.Module):
    def __init__(self, temperature, pooling=False, factor=1.0):
        super(Intra_sample_p2p_loss, self).__init__()
        self.temperature = temperature
        self.pooling = pooling
        self.factor = factor
    
    def pair_wise_sim_map(self, fea_0, fea_1):
        B, C, H, W = fea_0.size()

        fea_0 = fea_0.reshape(B, C, -1).transpose(1, 2)
        fea_1 = fea_1.reshape(B, C, -1)
        
        sim_map_0_1 = torch.matmul(fea_0, fea_1)
        sim_map_0_1 = torch.einsum('bic,dcj->bdij', fea_0, fea_1)
        return sim_map_0_1.reshape(-1, sim_map_0_1.shape[-1])


    def forward(self, feat_S, feat_T, type='fast'):
        B, C, H, W = feat_S.size()
        
        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)
        
        sim_dis = torch.tensor(0.).cuda()

        s_sim_map = self.pair_wise_sim_map(feat_S, feat_S)
        t_sim_map = self.pair_wise_sim_map(feat_T, feat_T)

        p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
        p_t = F.softmax(t_sim_map / self.temperature, dim=1)

        sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
        sim_dis += sim_dis_

        return sim_dis * self.factor


class Memory_p2p_loss(nn.Module):
    def __init__(self, pixel_memory_size=20000, region_memory_size=2000, region_contrast_size=1024, pixel_contrast_size=4096, 
                 contrast_kd_temperature=1.0, contrast_temperature=0.1, s_channels=1920, t_channels=1920, factor=0.1):
        super(Memory_p2p_loss, self).__init__()
        self.contrast_kd_temperature = contrast_kd_temperature
        self.contrast_temperature = contrast_temperature
        self.dim = t_channels

        self.project_head = nn.Sequential(
            nn.Conv2d(s_channels, t_channels, 1, bias=False),
            nn.SyncBatchNorm(t_channels),
            nn.ReLU(True),
            nn.Conv2d(t_channels, t_channels, 1, bias=False)
        )

        self.pixel_memory_size = pixel_memory_size
        self.pixel_update_freq = 128
        self.pixel_contrast_size = pixel_contrast_size

        self.factor = factor

        self.register_buffer("teacher_pixel_queue", torch.randn(self.pixel_memory_size, self.dim))
        self.teacher_pixel_queue = nn.functional.normalize(self.teacher_pixel_queue, p=2, dim=1)
        self.register_buffer("pixel_queue_ptr", torch.zeros(1, dtype=torch.long))

        float_indices = torch.linspace(0, self.pixel_memory_size - 1, self.pixel_contrast_size)
        rounded_indices = torch.round(float_indices)
        self.sampled_indices = rounded_indices.long()


    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

    
    def _dequeue_and_enqueue(self, keys):
        # segment_queue = self.teacher_segment_queue
        pixel_queue = self.teacher_pixel_queue

        keys = self.concat_all_gather(keys)
        
        batch_size, feat_dim, H, W = keys.size()

        this_feat = keys.contiguous().view(feat_dim, -1)

        # pixel enqueue and dequeue
        num_pixel = this_feat.shape[1]
        perm = torch.randperm(num_pixel)    
        K = min(num_pixel, self.pixel_update_freq)
        feat = this_feat[:, perm[:K]]
        feat = torch.transpose(feat, 0, 1)
        ptr = int(self.pixel_queue_ptr[0])

        if ptr + K >= self.pixel_memory_size:
            pixel_queue[-K:, :] = nn.functional.normalize(feat, p=2, dim=1)
            self.pixel_queue_ptr[0] = 0
        else:
            pixel_queue[ptr:ptr + K, :] = nn.functional.normalize(feat, p=2, dim=1)
            self.pixel_queue_ptr[0] = (self.pixel_queue_ptr[0] + K) % self.pixel_memory_size


    def contrast_sim_kd(self, s_logits, t_logits):
        p_s = F.log_softmax(s_logits/self.contrast_kd_temperature, dim=2)
        p_t = F.softmax(t_logits/self.contrast_kd_temperature, dim=2)
        sim_dis = F.kl_div(p_s, p_t, reduction='batchmean') * self.contrast_kd_temperature**2
        return sim_dis / p_s.shape[1]


    def forward(self, s_feats, t_feats):
        
        t_feats = F.normalize(t_feats, p=2, dim=1)
        s_feats = self.project_head(s_feats)
        s_feats = F.normalize(s_feats, p=2, dim=1)
        
        B, C, H, W = s_feats.shape

        self._dequeue_and_enqueue(t_feats.detach().clone())
        
        pixel_queue_size, feat_size = self.teacher_pixel_queue.shape
        t_X_pixel_contrast = self.teacher_pixel_queue[self.sampled_indices, :]

        t_feats = t_feats.reshape(B, C, -1).transpose(1, 2)
        s_feats = s_feats.reshape(B, C, -1).transpose(1, 2)
        t_pixel_logits = torch.div(torch.matmul(t_feats, t_X_pixel_contrast.T), self.contrast_temperature)
        s_pixel_logits = torch.div(torch.matmul(s_feats, t_X_pixel_contrast.T), self.contrast_temperature)

        pixel_sim_dis = self.contrast_sim_kd(s_pixel_logits, t_pixel_logits.detach()) * self.factor
        
        loss = pixel_sim_dis
        return loss

class RCFD(nn.Module):
    def __init__(self, temp=1.0):
        super(RCFD, self).__init__()
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, s_feats, t_feats):
        
        loss = F.kl_div(F.log_softmax(s_feats / self.temp, dim=-1), F.softmax(t_feats.detach(), dim=-1), reduction='batchmean')
        return loss
        
        