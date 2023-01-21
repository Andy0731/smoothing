# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.

import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
import torchvision.models as torchvision_models

from datasets import get_dataset
from torch.optim import SGD, Optimizer
import time
import datetime
from train_utils import AverageMeter, adjust_learning_rate, adjust_moco_momentum
from attrdict import AttrDict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from common import get_args
from archs.normal_resnet import resnet152 as normal_resnet152

# add for moco
import moco.optimizer
import moco.builder
import torch.backends.cudnn as cudnn
from functools import partial
import shutil


def main_spawn(args):
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.node_num
    mp.spawn(main_worker, nprocs=args.world_size, args=(args,))


def main_worker(gpu, args):
    args.global_rank = gpu
    args.local_rank = gpu % args.ngpus_per_node
    args.gpu = args.local_rank
    torch.cuda.set_device(args.local_rank)
    print('global_rank ', args.global_rank, ' local_rank ', args.local_rank, ' GPU ', torch.cuda.current_device())
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.global_rank)
    main(args)


def multinode_start(args, env_args):
    args.master_uri = "tcp://%s:%s" % (env_args.get("MASTER_ADDR"), env_args.get("MASTER_PORT"))
    args.node_rank = env_args.get("NODE_RANK")
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = env_args.get("WORLD_SIZE")
    args.local_rank = env_args.get("LOCAL_RANK")
    args.gpu = args.local_rank
    args.global_rank = env_args.get("RANK")
    torch.cuda.set_device(args.local_rank)
    print('global_rank ', args.global_rank, ' local_rank ', args.local_rank, ' GPU ', torch.cuda.current_device())
    dist.init_process_group(backend=args.dist_backend, init_method=args.master_uri, world_size=args.world_size, rank=args.global_rank)
    main(args)


def main(args):

    time_start = time.time()
    writer = SummaryWriter(args.outdir) if args.global_rank == 0 else None

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo_ResNet(
        eval(args.arch), args.moco_dim, args.moco_mlp_dim, args.moco_t)
    
    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch * args.world_size / 256    

    # apply SyncBN
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.local_rank)
    if args.ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    print(model)

    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    use_amp = True if (hasattr(args, 'amp') and args.amp) else False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    args.use_amp = use_amp

    # if retry
    retry_ckpt = os.path.join(args.retry_path, 'checkpoint.pth.tar')
    if os.path.isfile(retry_ckpt):
        args.resume = retry_ckpt

    args.start_epoch = 0

    # optionally resume from a checkpoint
    if hasattr(args, 'resume') and args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    dataaug = args.dataaug if hasattr(args, 'dataaug') else None
    train_dataset = get_dataset(args.dataset, 'train', args.data, dataaug, args.noise_sd)
    has_testset = False if (args.dataset == 'ti500k' or args.moco) else True
    if has_testset:
        test_dataset = get_dataset(args.dataset, 'test', args.data)

    # pin_memory = (args.dataset == "imagenet")
    pin_memory = True
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch,
            num_workers=args.workers, pin_memory=pin_memory, sampler=train_sampler, drop_last=True)
        if has_testset:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            test_loader = DataLoader(test_dataset, batch_size=args.batch,
                num_workers=args.workers, pin_memory=pin_memory, sampler=test_sampler)        
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)
        if has_testset:
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)

    # if hasattr(args, 'resume') and args.resume:
    #     resume_path = os.path.join(args.smoothing_path, args.resume)
    #     print('Loading checkpoint ', resume_path)
    #     assert os.path.isfile(resume_path), 'Could not find {}'.format(resume_path)
    #     ckpt = torch.load(resume_path)
    #     print('Checkpoint info: ', 'epoch ', ckpt['epoch'], ' arch ', ckpt['arch'])
    #     model_sd = ckpt['state_dict']
    #     # model_sd = {k[len('module.'):]:v for k,v in model_sd.items()}
    #     new_model_sd = {}
    #     for k,v in model_sd.items():
    #         if 'finetune' in args.outdir and ('fc' in k or 'linear' in k):
    #             continue 
    #         new_model_sd[k[len('module.'):]] = v
    #     model_sd = new_model_sd
    #     strict = False if 'finetune' in args.outdir else True        
    #     model.load_state_dict(model_sd, strict=strict)

    if args.ddp and args.global_rank == 0:
        print(args)

    for epoch in range(args.start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, optimizer, scaler, writer, epoch, args)

        if args.global_rank == 0:
            ckpt_file = os.path.join(args.outdir, 'checkpoint.pth.tar')
            ckpt_file_cp = os.path.join(args.retry_path, 'checkpoint.pth.tar')
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, ckpt_file)
                shutil.copyfile(ckpt_file, ckpt_file_cp)
            except OSError:
                print("OSError when saving checkpoint in epoch ", epoch)
    
    time_train_end = time.time()
    time_train = datetime.timedelta(seconds=time_train_end - time_start)
    print('training time: ', time_train)

   
# def init_seeds(seed=0, cuda_deterministic=True):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
#     if cuda_deterministic:  # slower, more reproducible
#         cudnn.deterministic = True
#         cudnn.benchmark = False
#     else:  # faster, less reproducible
#         cudnn.deterministic = False
#         cudnn.benchmark = True
    
def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for i, (images, _) in enumerate(train_loader):

        if args.debug and i > 1:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].size(0))
        if summary_writer:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("lr", lr, epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'lr: {lr:.4f}\t'
                    'GPU: {gpu}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, i, iters_per_epoch, batch_time=batch_time,
                data_time=data_time, loss=losses, gpu=args.global_rank, lr=lr))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

   
if __name__ == "__main__":

    env_args = get_args()

    parser = argparse.ArgumentParser(description='Robustness')
    parser.add_argument('--config', type=str, required=True)
    ps = parser.parse_args()
    cfg_file = os.path.basename(ps.config)
    cfg_dir = os.path.dirname(ps.config).split('/')[-1]
    cfg_file = os.path.join(cfg_dir, cfg_file)
    print('config: ', cfg_file)
    cfg = json.load(open(os.path.join("configs", cfg_file)))
    args = AttrDict(cfg)
    args.output = os.environ.get('AMLT_OUTPUT_DIR', os.path.join('/D_data/kaqiu/randomized_smoothing/', args.dataset))
    assert args.dataset in ['cifar10', 'imagenet', 'imagenet32', 'ti500k'], 'dataset must be cifar10 or imagenet or ti500k, but got {}'.format(args.dataset)
    if args.dataset == 'cifar10':
        args.data = os.environ.get('AMLT_DATA_DIR', '/D_data/kaqiu/cifar10/')
        if args.data == '/D_data/kaqiu/cifar10/': # local
            args.smoothing_path = '../amlt'
        else: # itp
            args.smoothing_path = args.data
    elif args.dataset == 'imagenet':
        args.data = os.environ.get('AMLT_DATA_DIR', '/D_data/kaqiu/imagenet/')
    elif args.dataset == 'imagenet32':
        args.data = os.environ.get('AMLT_DATA_DIR', '/D_data/kaqiu/imagenet32/')
    elif args.dataset == 'ti500k':
        args.data = os.environ.get('AMLT_DATA_DIR', '/D_data/kaqiu/ti500k/')

    if args.debug == 1:
        args.node_num = 1
        args.batch = min(8, args.batch)
        args.epochs = 100
        args.skip = 5000
        args.skip_train = 100000

    args.retry_path = os.path.join(args.data, 'smoothing', cfg_file.replace('.json',''))
    args.outdir = os.path.join(args.output, cfg_file.replace('.json', ''))
    if args.node_num > 1:
        if env_args.get('RANK') == 0 and not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        if env_args.get('RANK') == 0 and not os.path.exists(args.retry_path):
            os.makedirs(args.retry_path)        
        multinode_start(args, env_args)
    else:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        if not os.path.exists(args.retry_path):
            os.makedirs(args.retry_path)        
        if args.ddp:
            main_spawn(args)
        else:
            main(args)
