# evaluate a smoothed classifier on a dataset
import os
from datasets import get_num_classes
from core import Smooth
from time import time
import datetime


def merge_ctf_files(ctf_filename, args):
    with open(ctf_filename, 'w') as outfile:
        for i in range(args.world_size):
            fname = ctf_filename + '_rank' + str(i)
            with open(fname) as infile:
                outfile.write(infile.read())
    return


def run_certify(args, base_classifier, loader, split='test'):

    skip = args.skip if split == 'test' else args.skip_train

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    ctf_name = os.path.join(args.outdir, 'certify_sigma{}_{}'.format(args.sigma, split) + '_rank{}'.format(args.global_rank))
    args.cft_name = ctf_name
    ctf_file = open(ctf_name, 'w')
    if args.global_rank == 0:
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=ctf_file, flush=True)

    # iterate through the dataset
    for i, (x, label) in enumerate(loader):

        # only certify every args.skip examples, and stop after args.max examples
        if i % skip != 0:
            continue

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.certify_bs)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(i + args.global_rank, label, prediction, radius, correct, time_elapsed))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i + args.global_rank, label, prediction, radius, correct, time_elapsed), file=ctf_file, flush=True)

    ctf_file.close()

#need rewrite
# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description='Certify many examples')
#     parser.add_argument("dataset", choices=DATASETS, help="which dataset")
#     parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
#     parser.add_argument("sigma", type=float, help="noise hyperparameter")
#     parser.add_argument("outfile", type=str, help="output file")
#     parser.add_argument("--batch", type=int, default=1000, help="batch size")
#     parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
#     parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
#     parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
#     parser.add_argument("--N0", type=int, default=100)
#     parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
#     parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
#     args = parser.parse_args()

#     args.output = os.environ.get('AMLT_OUTPUT_DIR', os.path.join('/D_data/kaqiu/randomized_smoothing/', args.dataset))
#     args.data = os.environ.get('AMLT_DATA_DIR', '/D_data/kaqiu/cifar10/')
#     args.outdir = os.path.join(args.output, os.path.join(os.path.dirname(args.base_classifier).split('/')[-2:]))
#     if not os.path.exists(args.outdir):
#         os.makedirs(args.outdir)

#     # load the base classifier
#     checkpoint = torch.load(args.base_classifier) #need check!!!
#     base_classifier = get_architecture(checkpoint["arch"], args.dataset)
#     base_classifier.load_state_dict(checkpoint['state_dict'])

#     test_dataset = get_dataset(args.dataset, args.split, args.data)
#     pin_memory = (args.dataset == "imagenet")
#     if args.ddp:
#         test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
#         test_loader = DataLoader(test_dataset, batch_size=1,
#             num_workers=args.workers, pin_memory=pin_memory, sampler=test_sampler)   
#     else:
#         test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1,
#                                 num_workers=args.workers, pin_memory=pin_memory)
    
#     run_certify(args, base_classifier, test_loader)
