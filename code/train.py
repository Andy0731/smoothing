# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.

import argparse
import json
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset
from architectures import get_architecture
from torch.optim import SGD, Optimizer, AdamW, Adam
import time
import datetime
from train_utils import AverageMeter, accuracy, get_noise, adjust_learning_rate, mixup_data, mixup_criterion, add_fnoise, exp_fnoise, add_fnoise_chn, l2_dist, accuracy_per_class
from attrdict import AttrDict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from certify import run_certify, merge_ctf_files
from analyze import plot_curve
from common import get_args
from archs.hug_vit import get_hug_model, get_hug_vit
from DRM import DiffusionModel, get_timestep
import numpy as np



def main_spawn(args):
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.node_num
    mp.spawn(main_worker, nprocs=args.world_size, args=(args,))


def main_worker(gpu, args):
    args.global_rank = gpu
    args.local_rank = gpu % args.ngpus_per_node
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
    args.global_rank = env_args.get("RANK")
    torch.cuda.set_device(args.local_rank)
    print('global_rank ', args.global_rank, ' local_rank ', args.local_rank, ' GPU ', torch.cuda.current_device())
    dist.init_process_group(backend=args.dist_backend, init_method=args.master_uri, world_size=args.world_size, rank=args.global_rank)
    main(args)

def main(args):

    time_start = time.time()
    writer = SummaryWriter(args.outdir) if args.global_rank == 0 else None

    dataaug = args.dataaug if hasattr(args, 'dataaug') else None
    img_size = args.img_size if hasattr(args, 'img_size') else None
    train_dataset = get_dataset(args.dataset, 'train', args.data, dataaug, img_size)
    has_testset = False if args.dataset in ['ti500k', 'imagenet22k'] else True
    if has_testset:
        test_dataset = get_dataset(args.dataset, 'test', args.data, dataaug, img_size)
    # pin_memory = (args.dataset == "imagenet")
    pin_memory = True
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch,
            num_workers=args.workers, pin_memory=pin_memory, sampler=train_sampler)
        if has_testset:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch,
                num_workers=args.workers, pin_memory=pin_memory, sampler=test_sampler)        
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)
        if has_testset:
            test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)

    # build model
    if 'hug' in args.outdir:
        model = get_hug_model(args.arch)
    elif 'diffusion' in args.outdir:
        model = get_hug_vit(args.arch)
    elif hasattr(args, 'favg') and args.favg:
        assert hasattr(args, 'avgn_loc') and hasattr(args, 'avgn_num') and hasattr(args, 'fnoise_sd')
        model = get_architecture(args.arch, args.dataset, avgn_loc=args.avgn_loc, avgn_num=args.avgn_num)
    elif hasattr(args, 'nconv') and args.nconv:
        assert hasattr(args, 'avgn_num') and hasattr(args, 'fnoise_sd')
        model = get_architecture(args.arch, args.dataset, avgn_num=args.avgn_num)
    elif hasattr(args, 'noise_sd_embed') and args.noise_sd_embed:
        emb_scl = args.emb_scl if hasattr(args, 'emb_scl') else 1000
        emb_dim = args.emb_dim if hasattr(args, 'emb_dim') else 32
        model = get_architecture(args.arch, args.dataset, nemb_layer=args.nemb_layer, emb_scl=emb_scl, emb_dim=emb_dim)
    else:
        model = get_architecture(args.arch, args.dataset)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.local_rank)
    if args.ddp:
        if 'atp' in args.outdir or hasattr(args, 'train_layer'):
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        else:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, )

    # build diffusion model
    diffusion_model = None
    if hasattr(args, 'diffusion') and args.diffusion:
        diffusion_model_path = os.path.join(args.data, 'diffusion', args.diffusion_model)
        diffusion_model = DiffusionModel(diffusion_model_path)
        args.t = get_timestep(sigma=args.sigma, model=diffusion_model)

    # loss function
    criterion = CrossEntropyLoss().cuda()

    # optimizer
    if 'vit' in args.arch:
        if hasattr(args, 'optim') and args.optim == 'adam':
            optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
        else:
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.01)
    # freeze all layers except the train_layer
    elif hasattr(args, 'train_layer') and args.train_layer == 'linear':
        for name, param in model.named_parameters():
            if not 'linear' in name:
                param.requires_grad = False
        optimizer = SGD(model.module[1].linear.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    args.use_amp = True if (hasattr(args, 'amp') and args.amp) else False
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # if finetune from checkpoint
    if hasattr(args, 'resume') and args.resume:

        if args.local == 1:
            args.resume = os.path.join(args.smoothing_path, args.resume)
        else:
            resume_list = args.resume.split('/')
            resume_list.pop(1)
            args.resume = '/'.join(resume_list)
            args.resume = os.path.join(args.smoothing_path, args.resume)

    # if retry
    retry = 0
    retry_ckpt = os.path.join(args.retry_path, 'checkpoint.pth.tar')
    if os.path.isfile(retry_ckpt):
        args.resume = retry_ckpt
        retry = 1

    if hasattr(args, 'resume'):
        print('Loading checkpoint ', args.resume)
        assert os.path.isfile(args.resume), 'Could not find {}'.format(args.resume)

        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.local_rank)
        ckpt = torch.load(args.resume, map_location=loc)

        print('Checkpoint info: ', 'epoch ', ckpt['epoch'], ' arch ', ckpt['arch'])

        # retry
        if retry:
            args.start_epoch = ckpt['epoch']
            model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scaler.load_state_dict(ckpt['scaler'])            

        # finetune on CIFAR10, pretrain on ImageNet32/Ti500k with MOCO
        elif 'moco' in args.resume:
            # rename moco pre-trained keys
            linear_keyword = 'linear'
            state_dict = ckpt['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model.module[1].load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

        # finetune on CIFAR10, pretrain on ImageNet32/Ti500k with adding noise
        elif 'finetune' in args.outdir:
            state_dict = ckpt['state_dict']
            for k in list(state_dict.keys()):
                if ('fc' in k) or ('linear' in k):
                    del state_dict[k]
            model.load_state_dict(state_dict, strict=False)

        else:
            model.load_state_dict(ckpt['state_dict'])

        del ckpt

    if not hasattr(args, 'warmup_epochs'):
        args.warmup_epochs = max(1, args.epochs // 10)
    
    args.test_freq = args.test_freq if hasattr(args, 'test_freq') else max(1, args.epochs // 10)

    if args.ddp and args.global_rank == 0:
        print(args)

    start_epoch = args.start_epoch if hasattr(args, 'start_epoch') else 0
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, epoch, args.noise_sd, scaler, diffusion_model=diffusion_model)
        if has_testset and (epoch % args.test_freq == 0):
            test_loss, test_acc = test(args, test_loader, model, criterion, args.noise_sd, diffusion_model=diffusion_model)
            # reduce over all gpus
            test_acc_local = torch.tensor([test_acc]).cuda()
            dist.all_reduce(test_acc_local, op=dist.ReduceOp.SUM)
            test_acc_local /= args.world_size
            test_acc = test_acc_local

        if args.global_rank == 0:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('noise', args.cur_noise, epoch)
            if has_testset and (epoch % args.test_freq == 0):
                writer.add_scalar('test_loss', test_loss, epoch)
                writer.add_scalar('test_acc', test_acc, epoch)       

            # ckpt_file = os.path.join(args.outdir, 'checkpoint.pth.tar')
            ckpt_file_cp = os.path.join(args.retry_path, 'checkpoint.pth.tar')
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, ckpt_file_cp)
                # shutil.copyfile(ckpt_file, ckpt_file_cp)
            except OSError:
                print("OSError when saving checkpoint in epoch ", epoch)
    
    if has_testset:
        test_loss, test_acc = test(args, test_loader, model, criterion, args.noise_sd, diffusion_model=diffusion_model)
        test_acc_local = torch.tensor([test_acc]).cuda()
        print('rank ', args.global_rank, ', test_acc_local ', test_acc_local)
        dist.all_reduce(test_acc_local, op=dist.ReduceOp.SUM)
        test_acc_local /= args.world_size
        print('rank ', args.global_rank, ', test_acc_local ', test_acc_local)
        if args.global_rank == 0:        
            writer.add_scalar('test_loss', test_loss, args.epochs)
            writer.add_scalar('test_acc', test_acc_local, args.epochs)

    time_train_end = time.time()
    time_train = datetime.timedelta(seconds=time_train_end - time_start)
    print('training time: ', time_train)


    simgas = [args.sigma]
    if hasattr(args, 'multiple_sigma'):
        simgas = [float(sigma) for sigma in args.multiple_sigma.split(',')]
    for sigma in simgas:
        args.sigma = sigma
        # certify test set
        if has_testset and args.ddp and args.certify:
            print('begin to certify {} test set ...'.format(args.sigma))
            certify_loader = DataLoader(test_dataset, batch_size=1,
                num_workers=args.workers, pin_memory=pin_memory, sampler=test_sampler)
            certify_plot(args, model, certify_loader, 'test', writer, diffusion_model=diffusion_model)
            time_ctf_test_end = time.time()
            time_ctf_test = datetime.timedelta(seconds=time_ctf_test_end - time_train_end)
            print('certify test set time: ', time_ctf_test)
        time_ctf_test_end = time.time()

    for sigma in simgas:
        args.sigma = sigma        
        # certify training set
        if args.ddp and hasattr(args, 'cert_train') and args.cert_train:
            print('begin to certify {} training set ...'.format(args.sigma))
            certify_loader = DataLoader(train_dataset, batch_size=1,
                num_workers=args.workers, pin_memory=pin_memory, sampler=train_sampler)
            certify_loader.sampler.set_epoch(0)
            certify_plot(args, model, certify_loader, 'train', writer, diffusion_model=diffusion_model)
            time_ctf_train_end = time.time()
            time_ctf_train = datetime.timedelta(seconds=time_ctf_train_end - time_ctf_test_end)
            print('certify training set time: ', time_ctf_train)


def certify_plot(args, model, certify_loader, split, writer, diffusion_model=None):
    run_certify(args, model, certify_loader, split, writer, diffusion_model=diffusion_model)
    close_flag = torch.ones(1).cuda()
    print('rank ', args.global_rank, ', close_flag ', close_flag)
    dist.all_reduce(close_flag, op=dist.ReduceOp.SUM)
    print('rank ', args.global_rank, ', close_flag ', close_flag)
    if args.global_rank == 0:
        ctf_filename = args.cft_name.replace('_rank0', '')
        merge_ctf_files(ctf_filename, args)
        plot_curve(ctf_filename)


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
    
    
def train(args: AttrDict, loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float, scaler=None, diffusion_model=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    # use AverageMeter to record the accuracy for each class
    if hasattr(args, 'clean_class'):
        class_acc = [AverageMeter() for _ in range(10)]

    # switch to train mode
    model.train()
        
    adjust_learning_rate(optimizer, epoch, args)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        #print weights of the last layer
        print('conv1 weight', model.module[1].conv1.weight[0,0,:,:])
        print('fc weight', model.module[1].linear.weight)
        if args.debug and i > 0:
            break

        inputs = inputs.cuda()
        targets = targets.cuda()
        args.cur_noise = noise_sd

        if not args.natural_train:
            if args.clean_image:
                inputs_cln = inputs.clone().detach()
                targets_cln = targets.clone().detach()

            if hasattr(args, 'mixup') and args.mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.mixup_alpha)                

            # augment inputs with noise
            noise_sd = get_noise(epoch, args, inputs.size(0))
            if hasattr(args, 'noise_mode') and args.noise_mode == 'batch_random': # (N,)
                args.cur_noise = noise_sd[0]
            else: 
                args.cur_noise = noise_sd

            if hasattr(args, 'diffusion') and args.diffusion:
                acc_noise = args.accurate_noise if hasattr(args, 'accurate_noise') else 0
                inputs = diffusion_model(inputs, args.t, acc_noise, noise_sd)
            else:
                if hasattr(args, 'noise_mode') and args.noise_mode == 'batch_random': # (N,):
                    inputs = inputs + torch.randn_like(inputs, device='cuda') * torch.from_numpy(noise_sd.reshape(-1,1,1,1)).to(inputs.dtype).to(inputs.device)
                elif hasattr(args, 'clean_class'):
                    clean_classes = args.clean_class.split(',')
                    # list of str to list of int
                    clean_classes = [int(x) for x in clean_classes]
                    # print('clean_classes: ', clean_classes)
                    # tensor to numpy
                    targets_np = targets.cpu().numpy()
                    # print('targets_np: ', targets_np)
                    noise_mask = [0 if x in clean_classes else noise_sd for x in targets_np]
                    # print('noise_mask: ', noise_mask)
                    noise_mask = torch.from_numpy(np.array(noise_mask)).to(inputs.dtype).to(inputs.device)
                    inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_mask.reshape(-1,1,1,1)
                else:
                    inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
            if hasattr(args, 'resize_after_noise'):
                # inputs = torchvision.transforms.functional.resize(inputs, args.resize_after_noise)
                inputs = torch.nn.functional.interpolate(inputs, args.resize_after_noise, mode='bicubic')

            # expand (x + gnoise) with k fnoise, 
            if hasattr(args, 'favg') and args.favg:
                inputs = add_fnoise(inputs, args.fnoise_sd, args.avgn_num) # (b,c,h,w) -> (bn,c,h,w)
            elif hasattr(args, 'nconv') and args.nconv:
                inputs = add_fnoise_chn(inputs, args.fnoise_sd, args.avgn_num) # (b,3,h,w) -> (b,n3,h,w)
            elif hasattr(args, 'fexp') and args.fexp:
                assert hasattr(args, 'exp_noise') and hasattr(args, 'exp_num')
                inputs, targets = exp_fnoise(inputs, targets, args.exp_noise, args.exp_num)

            if args.clean_image:
                inputs = torch.cat((inputs_cln, inputs), dim=0)
                targets = torch.cat((targets_cln, targets), dim=0)

        # compute output
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            if hasattr(args, 'noise_sd_embed') and args.noise_sd_embed:
                outputs = model(inputs, noise_sd)
            else:
                outputs = model(inputs)
            if 'hug' in args.outdir or 'diffusion' in args.outdir:
                outputs = outputs.logits
            if hasattr(args, 'mixup') and args.mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if hasattr(args, 'clean_class'):
            acc_cls, num_cls, pred = accuracy_per_class(outputs, targets, num_classes=10)
            # print('targets: ', targets)
            # print('outputs top1: ', outputs.topk(1, dim=1)[1].squeeze())
            # print('outputs: ', outputs)
            # print('acc1: ', acc1)
            # print('acc_cls: ', acc_cls)
            for cls_ in range(len(acc_cls)):
                if num_cls[cls_] == 0:
                    continue
                class_acc[cls_].update(acc_cls[cls_], num_cls[cls_])
                # print('class ', i, ' acc: ', class_acc[i].avg)


        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.zero_grad() # set_to_none=True here can modestly improve performance

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'lr: {lr:.5f}\t'
                  'GPU: {gpu}\t'
                #   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Noise: {noise:.3f}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, 
                gpu=args.global_rank, lr=optimizer.param_groups[0]['lr'], noise=args.cur_noise))
            if hasattr(args, 'clean_class'):
                for cls_ in range(len(class_acc)):
                    print('class ', cls_, ' acc: ', class_acc[cls_].avg)

    return (losses.avg, top1.avg)


def test(args: AttrDict, loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float, diffusion_model=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # use AverageMeter to record the accuracy for each class
    if hasattr(args, 'clean_class'):
        class_acc = [AverageMeter() for _ in range(10)]

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.debug and i > 1:
                break

            inputs = inputs.cuda()
            targets = targets.cuda()

            if hasattr(args, 'diffusion') and args.diffusion:
                inputs = diffusion_model(inputs, args.t)
            elif hasattr(args, 'clean_class'):
                clean_classes = args.clean_class.split(',')
                # list of str to list of int
                clean_classes = [int(x) for x in clean_classes]
                # print('clean_classes: ', clean_classes)
                # tensor to numpy
                targets_np = targets.cpu().numpy()
                # print('targets_np: ', targets_np)
                noise_mask = [0 if x in clean_classes else noise_sd for x in targets_np]
                # print('noise_mask: ', noise_mask)
                noise_mask = torch.from_numpy(np.array(noise_mask)).to(inputs.dtype).to(inputs.device)
                inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_mask.reshape(-1,1,1,1)
            else:
                inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
            if hasattr(args, 'resize_after_noise'):
                # inputs = torchvision.transforms.functional.resize(inputs, args.resize_after_noise)
                inputs = torch.nn.functional.interpolate(inputs, args.resize_after_noise, mode='bicubic')

            # expand (x + gnoise) with k fnoise, 
            if hasattr(args, 'favg') and args.favg:
                inputs = add_fnoise(inputs, args.fnoise_sd, args.avgn_num) # (b,c,h,w) -> (bn,c,h,w)
            elif hasattr(args, 'nconv') and args.nconv:
                inputs = add_fnoise_chn(inputs, args.fnoise_sd, args.avgn_num) # (b,3,h,w) -> (b,n3,h,w)

            # compute output
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                if hasattr(args, 'noise_sd_embed') and args.noise_sd_embed:
                    outputs = model(inputs, noise_sd)
                else:
                    outputs = model(inputs)
                if 'hug' in args.outdir or 'diffusion' in args.outdir:
                    outputs = outputs.logits
                loss = criterion(outputs, targets)

            # measure accuracy and record loss            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if hasattr(args, 'clean_class'):
                acc_cls, num_cls, pred = accuracy_per_class(outputs, targets, num_classes=10)
                # print('target ', targets)
                # print('pred   ', pred)
                # print(acc_cls, num_cls)
                for cls_ in range(len(acc_cls)):
                    if num_cls[cls_] == 0:
                        continue
                    class_acc[cls_].update(acc_cls[cls_], num_cls[cls_])
                    # print('class ', cls_, ' acc: ', class_acc[cls_].avg)

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU {gpu}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5, gpu=args.global_rank))
                
                if hasattr(args, 'clean_class'):
                    for cls_ in range(len(class_acc)):
                        print('class ', cls_, ' acc: ', class_acc[cls_].avg)

        return (losses.avg, top1.avg)


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
    assert args.dataset in ['cifar10', 'imagenet', 'imagenet32', 'ti500k', 'imagenet22k'], 'dataset must be cifar10 or imagenet or ti500k, but got {}'.format(args.dataset)
    args.data = os.environ.get('AMLT_DATA_DIR', os.path.join('/D_data/kaqiu', args.dataset))

    if '/D_data/kaqiu' in args.data: # local
        args.local = 1
        args.smoothing_path = '../amlt'
    else: # itp
        args.local = 0
        args.smoothing_path = args.data

    if args.dataset == 'cifar10' and args.local == 0 and hasattr(args, 'pretrain_data'):
        args.smoothing_path = args.data.replace('cifar', args.pretrain_data)
        print('args.smoothing_path: ', args.smoothing_path)

    if args.debug == 1:
        args.node_num = 1
        args.batch = min(16, args.batch)
        args.epochs = 10
        args.skip = 10000
        args.skip_train = 200000
        args.N = 128
        args.certify_bs = 128
    
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
