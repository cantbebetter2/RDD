import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def make_beta_cosine(T, cosine_s=8e-3):
    timesteps = (torch.arange(T + 1, dtype=torch.float64) / T + cosine_s)
    alphas = timesteps / (1 + cosine_s) * math.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = betas.clamp(max=0.999)
    return betas
    

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, T, time_scale, loss_type, mean_type):
        super().__init__()

        self.model = model
        self.T = T
        self.time_scale = time_scale
        self.loss_type = loss_type
        self.mean_type = mean_type

        self.register_buffer(
            'betas', make_beta_cosine(self.T * self.time_scale + 1)) # change from T to T+1
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
    
    def predict_eps_from_x(self, z_t, x_0, t):
        # z_t = sqrt_alphas_bar * x_0 + sqrt_one_minus_alphas_bar * eps
        # -> eps = (z_t - sqrt_alphas_bar * x_0) / sqrt_one_minus_alphas_bar
        eps = (
            (z_t - extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0) / 
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape))
        return eps

    def forward(self, x_0, y=-1): # predict x_0 directly
        t = torch.randint(self.T + 1, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        z_t = (
            extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape) * noise)
        if self.mean_type == 'xstart':
            x_0_rec = self.model(z_t, t * self.time_scale, y) # x_hat in PD page 5
            bs = x_0.size(0)
            loss_x_0 = torch.mean(F.mse_loss(x_0_rec, x_0, reduction='none').reshape(bs, -1), dim=-1)
            loss_eps = torch.mean(F.mse_loss(self.predict_eps_from_x(z_t, x_0_rec, t * self.time_scale), noise, reduction='none').reshape(bs, -1), dim=-1)
            if self.loss_type == 'x':
                return torch.mean(loss_x_0)
            elif self.loss_type == 'eps':
                return torch.mean(loss_eps)
            elif self.loss_type == 'both':
                return torch.mean(torch.maximum(loss_x_0, loss_eps)) # truncated SNR weighting
        elif self.mean_type == 'epsilon':
            eps = self.model(z_t, t * self.time_scale, y)
            return F.mse_loss(eps, noise)


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, T, time_scale, img_size=32,
                 mean_type='xstart', var_type='fixedlarge', loss_type='both'):
        assert mean_type in ['xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.time_scale = time_scale
        self.loss_type = loss_type

        self.register_buffer(
            'betas', make_beta_cosine(T * time_scale + 1)) # change from T to T+1
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

    def predict_xstart_from_eps(self, z_t, eps, t):
        # z_t = sqrt_alphas_bar * x_0 + sqrt_one_minus_alphas_bar * eps
        # -> x_0 = (z_t - sqrt_one_minus_alphas_bar * eps) / sqrt_alphas_bar
        #        = (1 / sqrt_alphas_bar) * z_t - (sqrt_one_minus_alphas_bar / sqrt_alphas_bar) * eps
        #        = sqrt_recip_alphas_bar * z_t - sqrt_recipm1_alphas_bar * eps
        assert z_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, z_t.shape) * z_t -
            extract(self.sqrt_recipm1_alphas_bar, t, z_t.shape) * eps
        )
    
    def predict_eps_from_x(self, z_t, x_0, t):
        # z_t = sqrt_alphas_bar * x_0 + sqrt_one_minus_alphas_bar * eps
        # -> eps = (z_t - sqrt_alphas_bar * x_0) / sqrt_one_minus_alphas_bar
        eps = (
            (z_t - extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0) / 
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape))
        return eps
        
    def forward(self, z_t, t, y, feature=False):
        return self.model(z_t, t, y, feature)

    def ddim(self, x_T, stride, clip=True, y=-1): # DDIM sampler with large stride
        z_t = x_T
        for time_step in reversed(range(stride, self.T + 1, stride)):
            t = z_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            s = z_t.new_ones([x_T.shape[0], ], dtype=torch.long) * (time_step - stride)
            if self.mean_type == 'xstart':
                x_0 = self.model(z_t, t * self.time_scale, y)
                if clip:
                    x_0 = torch.clip(x_0, -1., 1.)
                eps = self.predict_eps_from_x(z_t, x_0, t * self.time_scale)
            elif self.mean_type == 'epsilon':
                eps = self.model(z_t, t * self.time_scale, y)
                x_0 = self.predict_xstart_from_eps(z_t, eps, t * self.time_scale)
                if clip:
                    x_0 = torch.clip(x_0, -1., 1.)
                    eps = self.predict_eps_from_x(z_t, x_0, t * self.time_scale)
            z_t = (
                extract(self.sqrt_alphas_bar, s * self.time_scale, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, s * self.time_scale, x_0.shape) * eps)
        return torch.clip(z_t, -1, 1)
    
    def PD(self, student, x_0, y=-1):
        t = 2 * torch.randint(1, student.module.T + 1, (x_0.shape[0],), device=x_0.device)
        # take teacher.T=512, student.T=256 for example, t \in 2 * [1, 2, ..., 256] = [2, 4, ..., 512]
        noise = torch.randn_like(x_0)
        with torch.no_grad():
            z_t = ( # start noise
                extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape) * noise) # create noise pictures, z_t in PD page 4
            if self.mean_type == 'xstart':
                x_0_rec = self.model(z_t, t * self.time_scale, y) # predicted x_0
                eps_rec = self.predict_eps_from_x(z_t, x_0_rec, t * self.time_scale) # predicted eps
            elif self.mean_type == 'epsilon':
                eps_rec = self.model(z_t, t * self.time_scale, y)
                x_0_rec = self.predict_xstart_from_eps(z_t, eps_rec, t * self.time_scale)
            z_t_minus_1 = (
                extract(self.sqrt_alphas_bar, (t - 1) * self.time_scale, x_0.shape) * x_0_rec +
                extract(self.sqrt_one_minus_alphas_bar, (t - 1) * self.time_scale, x_0.shape) * eps_rec) # get z_t' in PD page 4
            
            if self.mean_type == 'xstart':
                x_0_rec_rec = self.model(z_t_minus_1, (t - 1) * self.time_scale, y)
                eps_rec_rec = self.predict_eps_from_x(z_t_minus_1, x_0_rec_rec, (t - 1) * self.time_scale)
            elif self.mean_type == 'epsilon':
                eps_rec_rec = self.model(z_t_minus_1, (t - 1) * self.time_scale, y)
                x_0_rec_rec = self.predict_xstart_from_eps(z_t_minus_1, eps_rec_rec, (t - 1) * self.time_scale)
            z_t_minus_2 = (
                extract(self.sqrt_alphas_bar, (t - 2) * self.time_scale, x_0.shape) * x_0_rec_rec +
                extract(self.sqrt_one_minus_alphas_bar, (t - 2) * self.time_scale, x_0.shape) * eps_rec_rec) # get z_t'' in PD page 4

            frac = extract(self.sqrt_one_minus_alphas_bar, (t - 2) * self.time_scale, x_0.shape) / extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape)
            x_target = (z_t_minus_2 - frac * z_t) / (extract(self.sqrt_alphas_bar, (t - 2) * self.time_scale, x_0.shape) - frac * extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape))
            eps_target = self.predict_eps_from_x(z_t, x_target, t * self.time_scale)
        
        if self.mean_type == 'xstart':
            x_0_predicted = student(z_t, t * self.time_scale, y)
            eps_predicted = self.predict_eps_from_x(z_t, x_0_predicted, t * self.time_scale)
        elif self.mean_type == 'epsilon':
            eps_predicted = student(z_t, t * self.time_scale, y)
            x_0_predicted = self.predict_xstart_from_eps(z_t, eps_predicted, t * self.time_scale)


        bs = x_0.size(0)
        loss_x_0 = torch.mean(F.mse_loss(x_0_predicted, x_target, reduction='none').reshape(bs, -1), dim=-1)
        loss_eps = torch.mean(F.mse_loss(eps_predicted, eps_target, reduction='none').reshape(bs, -1), dim=-1)
        
        if self.loss_type == 'x':
            return torch.mean(loss_x_0)
        elif self.loss_type == 'eps':
            return torch.mean(loss_eps)
        elif self.loss_type == 'both':
            return torch.mean(torch.maximum(loss_x_0, loss_eps))
    
    def Entropy(self, logit):
        p = F.softmax(logit, dim=1)
        entropy = -(p * torch.log(p)).sum(dim=1)
        return entropy
    
    def Diversity(self, logit):
        p = F.softmax(logit, dim=1)
        p = p.mean(dim=0)
        diversity = (p * torch.log(p)).sum()
        return diversity

    def RCFD(self, student, classifier, x_0, y=-1, temp=0.9, alpha=0, beta=0, imagenet_cls=False):
        t = 2 * torch.randint(1, student.module.T + 1, (x_0.shape[0],), device=x_0.device)
        # take teacher.T=512, student.T=256 for example, t \in 2 * [1, 2, ..., 256] = [2, 4, ..., 512]
        noise = torch.randn_like(x_0)
        with torch.no_grad():
            z_t = ( # start noise
                extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape) * noise) # create noise pictures, z_t in PD page 4
            if self.mean_type == 'xstart':
                x_0_rec = self.model(z_t, t * self.time_scale, y) # predicted x_0
                eps_rec = self.predict_eps_from_x(z_t, x_0_rec, t * self.time_scale) # predicted eps
            elif self.mean_type == 'epsilon':
                eps_rec = self.model(z_t, t * self.time_scale, y)
                x_0_rec = self.predict_xstart_from_eps(z_t, eps_rec, t * self.time_scale)
            z_t_minus_1 = (
                extract(self.sqrt_alphas_bar, (t - 1) * self.time_scale, x_0.shape) * x_0_rec +
                extract(self.sqrt_one_minus_alphas_bar, (t - 1) * self.time_scale, x_0.shape) * eps_rec) # get z_t' in PD page 4
            
            if self.mean_type == 'xstart':
                x_0_rec_rec = self.model(z_t_minus_1, (t - 1) * self.time_scale, y)
                eps_rec_rec = self.predict_eps_from_x(z_t_minus_1, x_0_rec_rec, (t - 1) * self.time_scale)
            elif self.mean_type == 'epsilon':
                eps_rec_rec = self.model(z_t_minus_1, (t - 1) * self.time_scale, y)
                x_0_rec_rec = self.predict_xstart_from_eps(z_t_minus_1, eps_rec_rec, (t - 1) * self.time_scale)
            z_t_minus_2 = (
                extract(self.sqrt_alphas_bar, (t - 2) * self.time_scale, x_0.shape) * x_0_rec_rec +
                extract(self.sqrt_one_minus_alphas_bar, (t - 2) * self.time_scale, x_0.shape) * eps_rec_rec) # get z_t'' in PD page 4

            frac = extract(self.sqrt_one_minus_alphas_bar, (t - 2) * self.time_scale, x_0.shape) / extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape)
            x_target = (z_t_minus_2 - frac * z_t) / (extract(self.sqrt_alphas_bar, (t - 2) * self.time_scale, x_0.shape) - frac * extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape))
            
            if imagenet_cls:
                x_target = F.interpolate(x_target, size=224, mode='bilinear', align_corners=True)
            p_T, feat_T = classifier(x_target) # teacher prediction
        
        if self.mean_type == 'xstart':
            x_0_predicted = student(z_t, t * self.time_scale, y)
        elif self.mean_type == 'epsilon':
            eps_predicted = student(z_t, t * self.time_scale, y)
            x_0_predicted = self.predict_xstart_from_eps(z_t, eps_predicted, t * self.time_scale)
        
        if imagenet_cls:
            x_0_predicted = F.interpolate(x_0_predicted, size=224, mode='bilinear', align_corners=True)
        p_S, feat_S = classifier(x_0_predicted) # student prediction

        # entropy_loss = self.Entropy(p_S).mean()
        # div_loss = self.Diversity(p_S)
        feat_loss = F.kl_div(F.log_softmax(feat_S, dim=-1), F.softmax(feat_T / temp, dim=-1).detach(), reduction='batchmean')
        # loss = feat_loss + alpha * (beta * entropy_loss + (1 - beta) * div_loss)
        loss = feat_loss
        # return loss, entropy_loss, div_loss, feat_loss
        return loss, 0, 0, feat_loss
    

    def RDD(self, student, classifier, x_0, y=-1, loss_fun=None, gt=False, add_loss=None, logits=False, temp=1.0):
        t = 2 * torch.randint(1, student.module.T + 1, (x_0.shape[0],), device=x_0.device)
        # take teacher.T=512, student.T=256 for example, t \in 2 * [1, 2, ..., 256] = [2, 4, ..., 512]
        noise = torch.randn_like(x_0)
        with torch.no_grad():
            z_t = ( # start noise
                extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape) * x_0 +
                extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape) * noise) # create noise pictures, z_t in PD page 4
            if self.mean_type == 'xstart':
                x_0_rec = self.model(z_t, t * self.time_scale, y) # predicted x_0
                eps_rec = self.predict_eps_from_x(z_t, x_0_rec, t * self.time_scale) # predicted eps
            elif self.mean_type == 'epsilon':
                eps_rec = self.model(z_t, t * self.time_scale, y)
                x_0_rec = self.predict_xstart_from_eps(z_t, eps_rec, t * self.time_scale)
            z_t_minus_1 = (
                extract(self.sqrt_alphas_bar, (t - 1) * self.time_scale, x_0.shape) * x_0_rec +
                extract(self.sqrt_one_minus_alphas_bar, (t - 1) * self.time_scale, x_0.shape) * eps_rec) # get z_t' in PD page 4
            
            if self.mean_type == 'xstart':
                x_0_rec_rec = self.model(z_t_minus_1, (t - 1) * self.time_scale, y)
                eps_rec_rec = self.predict_eps_from_x(z_t_minus_1, x_0_rec_rec, (t - 1) * self.time_scale)
            elif self.mean_type == 'epsilon':
                eps_rec_rec = self.model(z_t_minus_1, (t - 1) * self.time_scale, y)
                x_0_rec_rec = self.predict_xstart_from_eps(z_t_minus_1, eps_rec_rec, (t - 1) * self.time_scale)
            z_t_minus_2 = (
                extract(self.sqrt_alphas_bar, (t - 2) * self.time_scale, x_0.shape) * x_0_rec_rec +
                extract(self.sqrt_one_minus_alphas_bar, (t - 2) * self.time_scale, x_0.shape) * eps_rec_rec) # get z_t'' in PD page 4

            frac = extract(self.sqrt_one_minus_alphas_bar, (t - 2) * self.time_scale, x_0.shape) / extract(self.sqrt_one_minus_alphas_bar, t * self.time_scale, x_0.shape)
            x_target = (z_t_minus_2 - frac * z_t) / (extract(self.sqrt_alphas_bar, (t - 2) * self.time_scale, x_0.shape) - frac * extract(self.sqrt_alphas_bar, t * self.time_scale, x_0.shape))
            eps_target = self.predict_eps_from_x(z_t, x_target, t * self.time_scale)
        
        if self.mean_type == 'xstart':
            x_0_predicted = student(z_t, t * self.time_scale, y)
            eps_predicted = self.predict_eps_from_x(z_t, x_0_predicted, t * self.time_scale)
        elif self.mean_type == 'epsilon':
            eps_predicted = student(z_t, t * self.time_scale, y)
            x_0_predicted = self.predict_xstart_from_eps(z_t, eps_predicted, t * self.time_scale)
        loss = 0.
        
        if logits:
            bs = x_0.size(0)
            loss_x_0 = torch.mean(F.mse_loss(x_0_predicted, x_target, reduction='none').reshape(bs, -1), dim=-1)
            loss_eps = torch.mean(F.mse_loss(eps_predicted, eps_target, reduction='none').reshape(bs, -1), dim=-1)
            
            logits_loss = 0.
            if self.loss_type == 'x':
                logits_loss += torch.mean(loss_x_0)
            elif self.loss_type == 'eps':
                logits_loss += torch.mean(loss_eps)
            elif self.loss_type == 'both':
                logits_loss += torch.mean(torch.maximum(loss_x_0, loss_eps))
            loss += logits_loss
        
        if classifier is not None:
            map_T, feat_T = classifier(x_target, type='mask')
            map_S, feat_S = classifier(x_0_predicted, type='mask')
            fun_loss = 0.
            add_loss_value = 0.
            if loss_fun is not None:
                fun_loss += loss_fun(map_S, map_T)
                loss += fun_loss
            if add_loss is not None:
                add_loss_value += add_loss(map_S, map_T)
                loss += add_loss_value
            rcdf_loss = F.kl_div(F.log_softmax(feat_S, dim=-1), F.softmax(feat_T / temp, dim=-1).detach(), reduction='batchmean')
            loss += rcdf_loss
        else:
            fun_loss = 0.
            add_loss_value = 0.
            rcdf_loss = 0.
            pass

        if gt:
            bs = x_0.size(0)
            
            loss_x_0 = torch.mean(F.mse_loss(x_0_rec, x_0, reduction='none').reshape(bs, -1), dim=-1)
            loss_eps = torch.mean(F.mse_loss(eps_predicted, noise, reduction='none').reshape(bs, -1), dim=-1)

            if self.loss_type == 'x':
                loss += torch.mean(loss_x_0)
            elif self.loss_type == 'eps':
                loss += torch.mean(loss_eps)
            elif self.loss_type == 'both':
                loss += torch.mean(torch.maximum(loss_x_0, loss_eps))
        output_content = f"total acc:{loss:.5f} | fun loss:{fun_loss:.5f} | rcdf loss:{rcdf_loss:.5f} | mini loss:{add_loss_value:.5f}"
        return loss, output_content

        