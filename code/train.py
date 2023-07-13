# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.

import argparse
import json
import os
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
import torch.nn.functional as F
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
import torchvision



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

    # loss function
    criterion = CrossEntropyLoss().cuda()

    # freeze all layers except the train_layer
    if hasattr(args, 'train_layer') and args.train_layer == 'linear':
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
        if hasattr(args, 'sep_cls_rbst') and args.sep_cls_rbst:
            train_loss, train_acc, kl_div, clean_loss = train(args, train_loader, model, criterion, optimizer, epoch, args.noise_sd, scaler, writer=writer)
        else:
            train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, epoch, args.noise_sd, scaler)
        
        if has_testset and (epoch % args.test_freq == 0 or epoch == args.epochs - 1):
            test_noise_sd = args.test_noise_sd if hasattr(args, 'test_noise_sd') else args.noise_sd
            if hasattr(args, 'test_mode'):
                for mode in args.test_mode:
                    test_loss, test_acc = test(args, test_loader, model, criterion, epoch, test_noise_sd, test_mode=mode)
            else:
                test_loss, test_acc = test(args, test_loader, model, criterion, epoch, test_noise_sd)
    
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
            if hasattr(args, 'sep_cls_rbst') and args.sep_cls_rbst:
                writer.add_scalar('kl_div', kl_div, epoch)
                writer.add_scalar('clean_loss', clean_loss, epoch)
       

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

                test_loss, test_acc = test(args, test_loader, model, criterion, epoch, test_noise_sd)
    
    if has_testset:
        test_loss, test_acc = test(args, test_loader, model, criterion, args.epochs, test_noise_sd, diffusion_model=diffusion_model)
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


def train(args: AttrDict, 
          loader: DataLoader, 
          model: torch.nn.Module, 
          criterion, 
          optimizer: Optimizer, 
          epoch: int, 
          noise_sd: float, 
          scaler=None, 
          diffusion_model=None,
          writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    # use AverageMeter to record the accuracy for each class
    if hasattr(args, 'acc_per_class') and args.acc_per_class:
        class_acc = [AverageMeter() for _ in range(10)]

    # switch to train mode
    model.train()
        
    adjust_learning_rate(optimizer, epoch, args)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        if args.debug and i > 0:
            break

        inputs = inputs.cuda()
        targets = targets.cuda()
        args.cur_noise = noise_sd
        # print('targets from data ', targets)

        # log the input images of the first batch using tensorboard writer
        if writer is not None and i == 0:
            img_grid = torchvision.utils.make_grid(inputs[:16], nrow=4)
            writer.add_image('input_images', img_grid, epoch)       

        # augment inputs with noise
        noise_sd = get_noise(epoch, args, inputs.size(0))
        if hasattr(args, 'noise_mode') and args.noise_mode == 'batch_random': # (N,)
            args.cur_noise = noise_sd[0]
        else: 
            args.cur_noise = noise_sd

        if hasattr(args, 'noise_mode') and args.noise_mode == 'batch_random': # (N,):
            inputs = inputs + torch.randn_like(inputs, device='cuda') * torch.from_numpy(noise_sd.reshape(-1,1,1,1)).to(inputs.dtype).to(inputs.device)
        elif hasattr(args, 'clean_class') and hasattr(args, 'sep_cls_rbst') and args.sep_cls_rbst:
            if hasattr(args, 'debug_sep') and args.debug_sep == 2: # clean 90, noise 10
                noise_inputs = inputs.clone().detach()
            elif hasattr(args, 'debug_sep') and (args.debug_sep == 3 or args.debug_sep == 5): # sep3: clean 70, noise 30; sep5: clean 90, noise 10
                noise_inputs = inputs.clone().detach()
                noise_inputs = noise_inputs + torch.randn_like(noise_inputs, device='cuda') * noise_sd
            elif hasattr(args, 'debug_sep') and args.debug_sep == 4: # sep4: clean 18, noise 18
                noise_inputs = torch.randn_like(inputs, device='cuda') * noise_sd
            else:
                clean_classes = args.clean_class.split(',')
                clean_classes = [int(x) if x else None for x in clean_classes]
                targets_np = targets.cpu().numpy()
                noise_mask = [0 if x in clean_classes else noise_sd for x in targets_np]
                noise_mask = torch.from_numpy(np.array(noise_mask)).to(inputs.dtype).to(inputs.device)
                noise_idx = torch.nonzero(noise_mask).squeeze()
                noise_inputs = inputs.clone().detach()[noise_idx]
                # print('noise_idx', noise_idx, ' noise_inputs', noise_inputs.shape)
                # if args.global_rank == 0 and i == 0:
                #     print('targets', targets)
                #     print('noise_idx', noise_idx)
                #     print('before, inputs', inputs[:,0,16,16])
                #     print('before, noise_inputs', noise_inputs[:,0,16,16])
                # noise_inputs = noise_inputs + torch.randn_like(noise_inputs, device='cuda') * noise_sd
                noise_inputs = noise_inputs + torch.randn_like(noise_inputs, device='cuda') * args.debug_noise_sd
                # if args.global_rank == 0 and i == 0:
                #     print('after, inputs', inputs[:,0,16,16])
                #     print('after, noise_inputs', noise_inputs[:,0,16,16])

            # log the input images of the first batch using tensorboard writer
            if writer is not None and i == 0:
                img_grid = torchvision.utils.make_grid(inputs[:16], nrow=4)
                writer.add_image('input_images_after_noise_mask', img_grid, epoch)
                noise_img_grid = torchvision.utils.make_grid(noise_inputs[:16], nrow=4)
                writer.add_image('noise_images', noise_img_grid, epoch)
        elif hasattr(args, 'clean_class'):
            clean_classes = args.clean_class.split(',')
            clean_classes = [int(x) for x in clean_classes]
            targets_np = targets.cpu().numpy()
            noise_mask = [0 if x in clean_classes else noise_sd for x in targets_np]
            noise_mask = torch.from_numpy(np.array(noise_mask)).to(inputs.dtype).to(inputs.device)
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_mask.reshape(-1,1,1,1)
            if writer is not None and i == 0:
                img_grid = torchvision.utils.make_grid(inputs[:16], nrow=4)
                writer.add_image('input_images_after_noise_mask', img_grid, epoch)
        else:
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
            if writer is not None and i == 0:
                img_grid = torchvision.utils.make_grid(inputs[:16], nrow=4)
                writer.add_image('input_images_after_noise_mask', img_grid, epoch)            

        if hasattr(args, 'resize_after_noise'):
            inputs = torch.nn.functional.interpolate(inputs, args.resize_after_noise, mode='bicubic')

        # compute output
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            if hasattr(args, 'clean_class') and hasattr(args, 'sep_cls_rbst') and args.sep_cls_rbst:
                outputs = model(inputs)
                if hasattr(args, 'debug_sep') and args.debug_sep == 5:
                    pass
                else:
                    noise_outputs = model(noise_inputs)
                clean_loss = criterion(outputs, targets)

                # print('targets cal loss ', targets)
                if hasattr(args, 'no_kl_loss') and args.no_kl_loss:
                    kl_div = torch.tensor(0.0).to(clean_loss.device)
                else:
                # get the kl divergence between the noisy outputs and clean outputs with noise_idx
                    kl_div = KLDivLoss(reduction='batchmean')(F.log_softmax(noise_outputs, dim=1), F.softmax(outputs[noise_idx], dim=1))
                loss = clean_loss + args.kl_weight * kl_div

                # log the input images of the first batch using tensorboard writer
                if writer is not None and i == 0:
                    img_grid = torchvision.utils.make_grid(inputs[:16], nrow=4)
                    writer.add_image('input_images_feed_to_model', img_grid, epoch)
                    noise_img_grid = torchvision.utils.make_grid(noise_inputs[:16], nrow=4)
                    writer.add_image('noise_images_feed_to_model', noise_img_grid, epoch)
    
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                if writer is not None and i == 0:
                    img_grid = torchvision.utils.make_grid(inputs[:16], nrow=4)
                    writer.add_image('input_images_feed_to_model', img_grid, epoch)               

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if hasattr(args, 'acc_per_class') and args.acc_per_class:
            acc_cls, num_cls, pred = accuracy_per_class(outputs, targets, num_classes=10)
            for cls_ in range(len(acc_cls)):
                if num_cls[cls_] == 0:
                    continue
                class_acc[cls_].update(acc_cls[cls_], num_cls[cls_])


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

    if args.global_rank == 0:
        print('Epoch: [{0}][{1}/{2}]\t'
                'lr: {lr:.5f}\t'
                'GPU: {gpu}\t'
            #   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Noise: {noise:.3f}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch, i+1, len(loader), batch_time=batch_time,
            data_time=data_time, loss=losses, top1=top1, top5=top5, 
            gpu=args.global_rank, lr=optimizer.param_groups[0]['lr'], noise=args.cur_noise))
        if hasattr(args, 'acc_per_class') and args.acc_per_class:
            for cls_ in range(len(class_acc)):
                print('class ', cls_, ' acc: ', class_acc[cls_].avg)
        if hasattr(args, 'sep_cls_rbst') and args.sep_cls_rbst:
            print('kl_div: ', kl_div.item(), ' clean_loss: ', clean_loss.item(), ' loss: ', loss.item())

    if hasattr(args, 'sep_cls_rbst') and args.sep_cls_rbst:
        return (losses.avg, top1.avg, kl_div.item(), clean_loss.item())
    return (losses.avg, top1.avg)


def test(args: AttrDict, loader: DataLoader, model: torch.nn.Module, criterion, epoch, noise_sd: float, diffusion_model=None, test_mode='all_noise'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # use AverageMeter to record the accuracy for each class
    if hasattr(args, 'acc_per_class') and args.acc_per_class:
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

            if hasattr(args, 'clean_class') and test_mode == 'mixcn':
                clean_classes = args.clean_class.split(',')
                clean_classes = [int(x) if x else None for x in clean_classes]
                targets_np = targets.cpu().numpy()
                noise_mask = [0 if x in clean_classes else noise_sd for x in targets_np]
                noise_mask = torch.from_numpy(np.array(noise_mask)).to(inputs.dtype).to(inputs.device)
                # if args.global_rank == 0 and i == 0:
                #     print('test on mixcn')
                #     print('targets', targets)
                #     print('noise_mask', noise_mask)
                #     print('before, inputs', inputs[:,0,16,16])
                inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_mask.reshape(-1,1,1,1)
                # if args.global_rank == 0 and i == 0:
                #     print('after, inputs', inputs[:,0,16,16])
            elif test_mode == 'all_noise':
                # if args.global_rank == 0 and i == 0:
                #     print('test on all noise')
                #     print('before, inputs', inputs[:,0,16,16])
                inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
                # if args.global_rank == 0 and i == 0:
                #     print('after, inputs', inputs[:,0,16,16])
            
            if hasattr(args, 'resize_after_noise'):
                # inputs = torchvision.transforms.functional.resize(inputs, args.resize_after_noise)
                inputs = torch.nn.functional.interpolate(inputs, args.resize_after_noise, mode='bicubic')

            # compute output
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # measure accuracy and record loss            
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if hasattr(args, 'acc_per_class') and args.acc_per_class:
                acc_cls, num_cls, pred = accuracy_per_class(outputs, targets, num_classes=10)
                for cls_ in range(len(acc_cls)):
                    if num_cls[cls_] == 0:
                        continue
                    class_acc[cls_].update(acc_cls[cls_], num_cls[cls_])

        if args.global_rank == 0:
            print('Test: [{epoch}]\t'
                    '{test_mode}\t'
                    'GPU {gpu}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch=epoch, loss=losses, top1=top1, top5=top5, gpu=args.global_rank, test_mode=test_mode))
            
            if hasattr(args, 'acc_per_class') and args.acc_per_class:
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
        args.epochs = 2
        args.skip = 10000
        args.skip_train = 200000
        args.N = 128
        args.certify_bs = 128

    args.acc_per_class = 1 if not hasattr(args, 'acc_per_class') else args.acc_per_class
    
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
