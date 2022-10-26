# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.

import argparse
import json
from multiprocessing import reduction
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset
from architectures import get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, LinearLR
import time
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log
from attrdict import AttrDict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from certify import run_certify, merge_ctf_files
from analyze import plot_curve


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


def main(args):

    writer = SummaryWriter(args.outdir) if args.global_rank == 0 else None

    train_dataset = get_dataset(args.dataset, 'train', args.data)
    test_dataset = get_dataset(args.dataset, 'test', args.data)
    # pin_memory = (args.dataset == "imagenet")
    pin_memory = True
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch,
            num_workers=args.workers, pin_memory=pin_memory, sampler=train_sampler)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=args.batch,
            num_workers=args.workers, pin_memory=pin_memory, sampler=test_sampler)        
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                                num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.local_rank)
    if args.ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.ddp and args.global_rank == 0:
        print(args)
        logfilename = os.path.join(args.outdir, 'log.txt')
        init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")

    criterion = CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler_step = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    if hasattr(args, 'warmup') and args.warmup == 1:
        scheduler_linear = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=4, last_epoch=- 1)

    for epoch in range(args.epochs):
        before = time.time()
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        if hasattr(args, 'consistency') and args.consistency:
            train_loss, train_acc, train_ce_loss, train_kl_loss, train_en_loss = train(args, train_loader, model, criterion, optimizer, epoch, args.noise_sd)
        else:
            train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, epoch, args.noise_sd)
        test_loss, test_acc = test(args, test_loader, model, criterion, args.noise_sd)
        after = time.time()
        if hasattr(args, 'warmup') and args.warmup == 1 and epoch < 5:
            scheduler_linear.step()
        scheduler_step.step()

        if args.global_rank == 0:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)
            if hasattr(args, 'consistency') and args.consistency:
                writer.add_scalar('train_ce_loss', train_ce_loss, epoch)
                writer.add_scalar('train_kl_loss', train_kl_loss, epoch)
                writer.add_scalar('train_en_loss', train_en_loss, epoch)

            log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                epoch, str(datetime.timedelta(seconds=(after - before))),
                scheduler_step.get_last_lr()[0], train_loss, train_acc, test_loss, test_acc))

            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'checkpoint.pth.tar'))
    
    if args.ddp and args.certify:
        certify_loader = DataLoader(test_dataset, batch_size=1,
            num_workers=args.workers, pin_memory=pin_memory, sampler=test_sampler)  
        run_certify(args, model, certify_loader)
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
    
    
def train(args: AttrDict, loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if hasattr(args, 'consistency') and args.consistency:
        ce_losses = AverageMeter()
        kl_losses = AverageMeter()
        en_losses = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.debug and i > 1:
            break

        inputs = inputs.cuda()
        targets = targets.cuda()

        if not args.natural_train:
            if args.clean_image:
                inputs_cln = inputs.clone().detach()
                targets_cln = targets.clone().detach()

            # augment inputs with noise
            if hasattr(args, 'consistency') and args.consistency:
                inputs = inputs.repeat(args.repeat_num, 1, 1, 1) # shape after repeat: (repeat_num * B, C, H, W)
                targets = targets.repeat(args.repeat_num)
            
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            if args.clean_image:
                inputs = torch.cat((inputs_cln, inputs), dim=0)
                targets = torch.cat((targets_cln, targets), dim=0)

        # compute output
        outputs = model(inputs) # (B, class_num)
        loss = criterion(outputs, targets)

        if hasattr(args, 'consistency') and args.consistency:
            c_batch = inputs.size(0) // args.repeat_num
            ce_loss = loss.clone().detach()
            probs = F.softmax(outputs, -1) # (repeat_num*B, 10)
            assert probs.shape == (args.repeat_num*c_batch, 10), 'outputs.shape: {}'.format(probs.shape)
            probs = probs.view(args.repeat_num, c_batch, 10) # (repeat_num, B, 10)
            probs_avg = torch.mean(probs, dim=0, keepdim=False) # (B, 10)
            probs_avg = probs_avg.repeat(args.repeat_num, 1) # (repeat_num*B, 10)
            probs = probs.view(args.repeat_num*c_batch, 10) # (repeat_num*B, 10)
            # print('probs: ', probs)
            # print('probs_avg: ', probs_avg)
            kl_loss = F.kl_div(probs, probs_avg, reduction='none', log_target=True).sum(-1).mean() * args.kl_loss_w
            # print('kl_loss: ', kl_loss.item())
            en_loss = (- probs * torch.log(probs)).sum(-1).mean() * args.en_loss_w
            if args.kl_loss:
                loss += kl_loss
            if args.en_loss:
                loss += en_loss
            ce_losses.update(ce_loss.item(), inputs.size(0))
            kl_losses.update(kl_loss.item(), inputs.size(0))
            en_losses.update(en_loss.item(), inputs.size(0))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'lr: {lr:.3f}\t'
                  'GPU: {gpu}\t'
                #   'Time {batch_time.avg:.3f}\t'
                #   'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'CE_Loss {ce_loss.avg:.4f}\t'
                  'KL_Loss {kl_loss.avg:.8f}\t'
                  'EN_Loss {en_loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, 
                gpu=args.global_rank, lr=optimizer.param_groups[0]['lr'],
                ce_loss=ce_losses, kl_loss=kl_losses, en_loss=en_losses))

    if hasattr(args, 'consistency') and args.consistency:
        return (losses.avg, top1.avg, ce_losses.avg, kl_losses.avg, en_losses.avg)
    return (losses.avg, top1.avg)


def test(args: AttrDict, loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

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

            if not args.natural_train:
                # augment inputs with noise
                inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
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

        return (losses.avg, top1.avg)


if __name__ == "__main__":

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
    args.data = os.environ.get('AMLT_DATA_DIR', '/D_data/kaqiu/cifar10/')
    args.outdir = os.path.join(args.output, cfg_file.replace('.json', ''))
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    if args.debug == 1:
        args.batch = min(64, args.batch)
        args.epochs = 1
        args.skip = 1000

    if args.ddp:
        main_spawn(args)
    else:
        main(args)
