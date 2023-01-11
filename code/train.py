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
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, LinearLR, CosineAnnealingLR
import time
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log, get_noise
from attrdict import AttrDict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from certify import run_certify, merge_ctf_files
from analyze import plot_curve
from common import get_args


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
    train_dataset = get_dataset(args.dataset, 'train', args.data, dataaug)
    has_testset = False if args.dataset == 'ti500k' else True
    if has_testset:
        test_dataset = get_dataset(args.dataset, 'test', args.data)
    # pin_memory = (args.dataset == "imagenet")
    pin_memory = True
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch,
            num_workers=args.workers, pin_memory=pin_memory, sampler=train_sampler)
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

    model = get_architecture(args.arch, args.dataset)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if hasattr(args, 'resume') and args.resume:
        resume_path = os.path.join(args.smoothing_path, args.resume)
        print('Loading checkpoint ', resume_path)
        assert os.path.isfile(resume_path), 'Could not find {}'.format(resume_path)
        ckpt = torch.load(resume_path)
        print('Checkpoint info: ', 'epoch ', ckpt['epoch'], ' arch ', ckpt['arch'])
        model_sd = ckpt['state_dict']
        # model_sd = {k[len('module.'):]:v for k,v in model_sd.items()}
        new_model_sd = {}
        for k,v in model_sd.items():
            if 'finetune' in args.outdir and ('fc' in k or 'linear' in k):
                continue 
            new_model_sd[k[len('module.'):]] = v
        model_sd = new_model_sd
        strict = False if 'finetune' in args.outdir else True        
        model.load_state_dict(model_sd, strict=strict)

    model.cuda(args.local_rank)
    if args.ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if args.ddp and args.global_rank == 0:
        print(args)
        logfilename = os.path.join(args.outdir, 'log.txt')
        init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")

    criterion = CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if hasattr(args, 'consin_lr') and args.consin_lr == 1:
        scheduler_step = CosineAnnealingLR(optimizer, args.epochs)
    else: 
        scheduler_step = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    if hasattr(args, 'warmup') and args.warmup == 1:
        scheduler_linear = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=4, last_epoch=- 1)

    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]['lr']
        before = time.time()
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_acc = train(args, train_loader, model, criterion, optimizer, epoch, args.noise_sd)
        if has_testset and (epoch % args.test_freq == 0 or epoch == args.epochs - 1):
            test_loader.sampler.set_epoch(epoch)
            test_loss, test_acc = test(args, test_loader, model, criterion, args.noise_sd)
        after = time.time()

        if args.global_rank == 0:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('noise', args.cur_noise, epoch)
            if has_testset and (epoch % args.test_freq == 0 or epoch == args.epochs - 1):
                writer.add_scalar('test_loss', test_loss, epoch)
                writer.add_scalar('test_acc', test_acc, epoch)

            if has_testset:
                log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                    epoch, str(datetime.timedelta(seconds=(after - before))),
                    lr, train_loss, train_acc, test_loss, test_acc))
            else:
                log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
                    epoch, str(datetime.timedelta(seconds=(after - before))),
                    lr, train_loss, train_acc))                

            if epoch == args.epochs - 1:
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(args.outdir, 'checkpoint.pth.tar'))
        
        if hasattr(args, 'warmup') and args.warmup == 1 and epoch < 5:
            scheduler_linear.step()
        else:
            scheduler_step.step()
    time_train_end = time.time()
    time_train = datetime.timedelta(seconds=time_train_end - time_start)
    print('training time: ', time_train)

    if args.ddp and hasattr(args, 'cert_train') and args.cert_train:
        certify_loader = DataLoader(train_dataset, batch_size=1,
            num_workers=args.workers, pin_memory=pin_memory, sampler=train_sampler)
        certify_loader.sampler.set_epoch(0)
        certify_plot(args, model, certify_loader, 'train')
        time_ctf_train_end = time.time()
        time_ctf_train = datetime.timedelta(seconds=time_ctf_train_end - time_train_end)
        print('certify training set time: ', time_ctf_train)

    if args.ddp and args.certify:
        certify_loader = DataLoader(test_dataset, batch_size=1,
            num_workers=args.workers, pin_memory=pin_memory, sampler=test_sampler)
        certify_loader.sampler.set_epoch(0)
        certify_plot(args, model, certify_loader, 'test')
        time_ctf_test_end = time.time()
        time_ctf_test = datetime.timedelta(seconds=time_ctf_test_end - time_ctf_train_end)
        print('certify test set time: ', time_ctf_test)

def certify_plot(args, model, certify_loader, split):
    run_certify(args, model, certify_loader, split)
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
    end = time.time()

    use_amp = True if (hasattr(args, 'amp') and args.amp) else False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        if args.debug and i > 1:
            break

        inputs = inputs.cuda()
        targets = targets.cuda()
        args.cur_noise = noise_sd

        if not args.natural_train:
            if args.clean_image:
                inputs_cln = inputs.clone().detach()
                targets_cln = targets.clone().detach()

            # augment inputs with noise
            noise_sd = get_noise(epoch, args)
            args.cur_noise = noise_sd
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            if args.clean_image:
                inputs = torch.cat((inputs_cln, inputs), dim=0)
                targets = torch.cat((targets_cln, targets), dim=0)

        # compute output
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

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
                  'lr: {lr:.3f}\t'
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

    return (losses.avg, top1.avg)


def test(args: AttrDict, loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    use_amp = True if (hasattr(args, 'amp') and args.amp) else False

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

            if hasattr(args, 'natural_test') and (not args.natural_test):
                # augment inputs with noise
                inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
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

    args.outdir = os.path.join(args.output, cfg_file.replace('.json', ''))
    if args.node_num > 1:
        if env_args.get('RANK') == 0 and not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        multinode_start(args, env_args)
    else:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        if args.debug == 1:
            args.batch = min(8, args.batch)
            args.epochs = 100
            args.skip = 5000
            args.skip_train = 100000

        if args.ddp:
            main_spawn(args)
        else:
            main(args)
