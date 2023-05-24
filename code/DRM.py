import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import sys
from train_utils import l2_dist


from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)



_VITCF10_MEAN = [0.5, 0.5, 0.5]
_VITCF10_STDDEV = [0.5, 0.5, 0.5]


def show(img, name, iter):
    img = img.cpu().detach()
    plt.figure()
    img = F.to_pil_image(img)
    plt.imshow(img)
    plt.savefig('cifar10/output/img' + str(iter) + '_' + name + '.pdf')


def get_timestep(sigma, model):
    target_sigma = sigma * 2
    real_sigma = 0
    t = 0
    while real_sigma < target_sigma:
        t += 1
        a = model.diffusion.sqrt_alphas_cumprod[t]
        b = model.diffusion.sqrt_one_minus_alphas_cumprod[t]
        real_sigma = b / a    
    return t

class Args:
    image_size=32
    num_channels=128
    num_res_blocks=3
    num_heads=4
    num_heads_upsample=-1
    attention_resolutions="16,8"
    dropout=0.3
    learn_sigma=True
    sigma_small=False
    class_cond=False
    diffusion_steps=4000
    noise_schedule="cosine"
    timestep_respacing=""
    use_kl=False
    predict_xstart=False
    rescale_timesteps=True
    rescale_learned_sigmas=True
    use_checkpoint=False
    use_scale_shift_norm=True


class DiffusionModel(nn.Module):
    def __init__(self, model_ckp):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load(model_ckp)
        )
        model.eval().cuda()

        self.model = model 
        self.diffusion = diffusion 

    def forward(self, x, t, acc_noise=0, noise_sd=0.25):
        if acc_noise > 0:
            noise = torch.randn_like(x) * noise_sd
            x_in = x + noise
            imgs = self.acc_denoise(x_in, t)
            return imgs
        else:
            x_in = x * 2 -1
            imgs = self.denoise(x_in, t)
            return imgs
        
    def acc_denoise(self, x_start, t):
        x_start = x_start * 2 - 1
        t_batch = torch.tensor([t] * len(x_start)).cuda()
        out = self.diffusion.p_sample(
            self.model,
            x_start,
            t_batch,
            clip_denoised=True
        )['pred_xstart']
        return out

    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    t_batch = torch.tensor([i] * len(x_start)).cuda()
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out