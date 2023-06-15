import torch
import math
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()

def log(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()

def get_noise(epoch, args, cur_bs=None):
    if hasattr(args, 'noise_mode') and args.noise_mode == 'li_inc':
        return args.noise_sd * float(epoch + 1) / float(args.epochs)
    elif hasattr(args, 'noise_mode') and args.noise_mode == 'li_dec':
        return args.noise_sd * (2 - float(epoch + 1) / float(args.epochs))
    elif hasattr(args, 'noise_mode') and args.noise_mode == 'li_inc_rand':
        low = args.noise_sd * float(epoch + 1) / float(args.epochs)
        return np.random.uniform(low, args.noise_sd)
    elif hasattr(args, 'noise_mode') and args.noise_mode == 'li_dec_rand':
        high = args.noise_sd * (2 - float(epoch + 1) / float(args.epochs))
        return np.random.uniform(args.noise_sd, high)
    elif hasattr(args, 'noise_mode') and args.noise_mode == 'li_range_rand':
        low = args.noise_sd * float(epoch + 1) / float(args.epochs)
        high = args.noise_sd * (2 - float(epoch + 1) / float(args.epochs))
        return np.random.uniform(low, high)
    elif hasattr(args, 'noise_mode') and args.noise_mode == 'linear':
        if epoch < args.noise_ep:
            return args.noise_sd * float(epoch) / float(args.noise_ep)
        else:
            return args.noise_sd
    elif hasattr(args, 'noise_mode') and args.noise_mode.startswith('step'):
        step = int(args.noise_mode.replace('step', ''))
        if epoch < args.noise_ep:
            return (epoch // (args.noise_ep // step)) * (args.noise_sd / step)
        else:
            return args.noise_sd
    elif hasattr(args, 'noise_mode') and args.noise_mode == 'random':
        random_scale = args.random_scale if hasattr(args, 'random_scale') else 1.2
        return np.random.uniform(0, args.noise_sd * random_scale)
    elif hasattr(args, 'noise_mode') and args.noise_mode == 'batch_random':
        random_scale = args.random_scale if hasattr(args, 'random_scale') else 1.2
        return np.random.uniform(0, args.noise_sd * random_scale, size=(cur_bs))        
    elif hasattr(args, 'noise_mode') and args.noise_mode == 'random_range':
        return np.random.uniform(args.noise_sd * args.r_range_min, args.noise_sd * args.r_range_max)
    else:
        return args.noise_sd

def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * (epoch + 1) / args.warmup_epochs               
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def avg_input_noise(x, noise_sd, num):
    b, c, h, w = x.shape
    nx = x.repeat_interleave(num, dim=0) #bn,c,h,w
    noise = torch.randn_like(nx, device='cuda') * noise_sd #bn,c,h,w
    noise = noise.view(b, num, c, h, w) #b,n,c,h,w
    noise = noise.mean(dim=1) #b,c,h,w
    del nx
    return noise

def add_fnoise(x, noise_sd, num):
    nx = x.repeat_interleave(num, dim=0) #bn, c, h, w
    noise = torch.randn_like(nx, device='cuda') * noise_sd
    return nx + noise

def exp_fnoise(inputs, targets, noise_sd, num):
    targets = targets.repeat_interleave(num, dim=0) # b -> bn
    inputs = inputs.repeat_interleave(num, dim=0) # b, c, h, w -> bn, c, h, w
    noise = torch.randn_like(inputs, device='cuda') * noise_sd
    inputs = inputs + noise
    return inputs, targets

def add_fnoise_chn(x, noise_sd, num): # (b,3,h,w) -> (b,n3,h,w)
    nx = x.repeat(1,num,1,1) #b,c,h,w -> b,nc,h,w
    noise = torch.randn_like(nx, device='cuda') * noise_sd
    return nx + noise    

def l2_dist(x, y): # (b,c,h,w) (b,c,h,w)
    n = y - x # (b,c,h,w)
    n = n.view(n.shape[0], -1) # (b,c*h*w)
    print('l2 norm of diffusion noise: ')
    print(torch.norm(n, dim=1)) # (b,)
    print('l2 norm of 0.5 noise: ')
    n025 = torch.randn_like(x, device='cuda') * 0.5
    n025 = n025.view(n025.shape[0], -1)
    print(torch.norm(n025, dim=1))