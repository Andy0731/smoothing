import os
import numpy as np
import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math

# get params from cli
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', type=str, default=None, help='path to ctf file')

args = parser.parse_args()
    


sns.set()


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> tuple:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        sv = [0.0, 0.25, 0.5, 0.75, 1.0]
        robust_acc_np, spe_robust_acc_np = self.get_curve(df=df, radii=radii, sv=sv, curve=self.at_radius)
        robust_np, spe_robust_np = self.get_curve(df=df, radii=radii, sv=sv, curve=self.at_radius_only_roubst)
        assert len(sv) == len(spe_robust_acc_np)
        assert len(sv) == len(spe_robust_np)

        return (np.array(robust_acc_np),
            np.array(robust_np),
            df["correct"].mean(),
            sv,
            spe_robust_acc_np,
            spe_robust_np)

    def get_curve(self, df: pd.DataFrame, radii: np.ndarray, sv: list, curve):
        rnp = []
        srnp = []
        for radius in radii:
            r = curve(df, radius)
            rnp.append(r)
            if radius in sv:
                srnp.append(r)
        return rnp, srnp

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()

    def at_radius_only_roubst(self, df: pd.DataFrame, radius: float):
        return (df["radius"] >= radius).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))


class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    # colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    # assert len(lines) < 8, 'plot at most 7*2 curves!'
    df_list = []
    exp_list = []
    for idx, line in enumerate(lines):
        robust_acc, robustness, accuracy, sv, spe_robust_acc_np, spe_robust_np  = line.quantity.at_radii(radii)
        # line.legend = line.legend + '_acc{:.1f}'.format(accuracy * 100)
        plt.plot(radii * line.scale_x, robust_acc, linestyle='solid', label=line.legend)
        # plt.plot(radii * line.scale_x, robust_acc, color=colors[idx % 7], linestyle='solid', label=line.legend)
        # plt.plot(radii * line.scale_x, robustness, color=colors[idx % 7], linestyle='dashed', label=line.legend + '_ro')
        df_list.append(spe_robust_acc_np)
        exp_list.append(line.legend + '_ra')
    
    exps = pd.DataFrame(np.array(df_list) * 100.0, index=exp_list, columns=sv)
    print(exps)
    exps.to_csv(outfile + '_curve.csv')
        
    # plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    # plt.tick_params(labelsize=14)
    # write a italic in latex
    plt.xlabel("Radius", fontsize=14)
    plt.ylabel("Certified accuracy", fontsize=14)
    plt.xlabel("Radius ($\ell_2$)", fontsize=14)
    plt.ylabel("Certified accuracy")
    # plt.xlabel("Radius (l2)", fontsize=16)
    # plt.ylabel("Certified accuracy", fontsize=16)
    # plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    # plt.legend([method.legend for method in lines], fontsize=10)
    # plt.legend(fontsize=6)
    plt.legend()
    # plt.grid(False)
    # plt.gca().set_facecolor('none')    
    plt.savefig(outfile + "_curve.pdf")
    plt.tight_layout()
    # plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(outfile + "_curve.png", dpi=300)
    plt.close()

    # plt.figure()
    # # colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    # # assert len(lines) < 8, 'plot at most 7*2 curves!'
    # df_list = []
    # exp_list = []    
    # for idx, line in enumerate(lines):
    #     robust_acc, robustness, accuracy, sv, spe_robust_acc_np, spe_robust_np = line.quantity.at_radii(radii)
    #     # line.legend = line.legend + '_acc{:.1f}'.format(accuracy * 100)
    #     # plt.plot(radii * line.scale_x, robust_acc, color=colors[idx % 7], linestyle='solid', label=line.legend)
    #     # plt.plot(radii * line.scale_x, robustness, color=colors[idx % 7], linestyle='dashed', label=line.legend + '_ro')
    #     plt.plot(radii * line.scale_x, robustness, linestyle='dashed', label=line.legend + '_r')
    #     df_list.append(spe_robust_np)
    #     exp_list.append(line.legend + '_r')

    # exps = pd.DataFrame(np.array(df_list) * 100.0, index=exp_list, columns=sv)
    # print(exps)
    # exps.to_csv(outfile + '_r.csv')
        
    # plt.ylim((0, 1))
    # plt.xlim((0, max_radius))
    # plt.tick_params(labelsize=14)
    # plt.xlabel("radius", fontsize=16)
    # plt.ylabel("certified accuracy", fontsize=16)
    # # plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    # # plt.legend([method.legend for method in lines], fontsize=10)
    # plt.legend(fontsize=10)
    
    # plt.savefig(outfile + "_r.pdf")
    # plt.tight_layout()
    # plt.title(title, fontsize=16)
    # plt.tight_layout()
    # plt.savefig(outfile + "_r.png", dpi=300)
    # plt.close()    


def smallplot_certified_accuracy(outfile: str, title: str, max_radius: float,
                                 methods: List[Line], radius_step: float = 0.01, xticks=0.5) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for method in methods:
        plt.plot(radii, method.quantity.at_radii(radii), method.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.xlabel("radius", fontsize=22)
    plt.ylabel("certified accuracy", fontsize=22)
    plt.tick_params(labelsize=20)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(xticks))
    plt.legend([method.legend for method in methods], loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.close()


def latex_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')

    for radius in radii:
        f.write("& $r = {:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    for i, method in enumerate(methods):
        f.write(method.legend)
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = r" & \textbf{" + "{:.2f}".format(accuracies[i, j]) + "}"
            else:
                txt = " & {:.2f}".format(accuracies[i, j])
            f.write(txt)
        f.write("\\\\\n")
    f.close()


def markdown_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                      methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')
    f.write("|  | ")
    for radius in radii:
        f.write("r = {:.3} |".format(radius))
    f.write("\n")

    f.write("| --- | ")
    for i in range(len(radii)):
        f.write(" --- |")
    f.write("\n")

    for i, method in enumerate(methods):
        f.write("<b> {} </b>| ".format(method.legend))
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = "{:.2f}<b>*</b> |".format(accuracies[i, j])
            else:
                txt = "{:.2f} |".format(accuracies[i, j])
            f.write(txt)
        f.write("\n")
    f.close()

def plot_curve(ctf_filename):
    lines = []
    exp = ctf_filename.split('/')[-3]
    setting = ctf_filename.split('/')[-1].split('_')[-1]
    if os.path.isfile(ctf_filename):
        lines.append(Line(ApproximateAccuracy(ctf_filename), exp + '_' + setting))
    plot_certified_accuracy(ctf_filename, exp, 1.0, lines)

if __name__ == "__main__":
    ctf_files = [

        # IN32 Pretrain with equal probability (0,0.25,0.5,1.0), CF100 Finetune on clean images
        # "amlt/smoothing/nep_n025n05n1_cln_cls_kl1_bfc_fzc_r152gn1_in32bs128_lr01e400_ft_cifar100_lr001e100/cifar100/nep_n025n05n1_cln_cls_kl1_bfc_fzc_r152gn1_in32bs128_lr01e400_ft_cifar100_lr001e100/certify_sigma0.25_test",
        # "amlt/smoothing/nep_n025n05n1_cln_cls_kl1_bfc_fzc_r152gn1_in32bs128_lr01e400_ft_cifar100_lr001e100/cifar100/nep_n025n05n1_cln_cls_kl1_bfc_fzc_r152gn1_in32bs128_lr01e400_ft_cifar100_lr001e100/certify_sigma0.5_test",
        # "amlt/smoothing/nep_n025n05n1_cln_cls_kl1_bfc_fzc_r152gn1_in32bs128_lr01e400_ft_cifar100_lr001e100/cifar100/nep_n025n05n1_cln_cls_kl1_bfc_fzc_r152gn1_in32bs128_lr01e400_ft_cifar100_lr001e100/certify_sigma1.0_test",

        # # IN32 pretrain with equal probability (0,0.25,0.5,1.0), CF10 Finetune on clean images lr001e100
        # "amlt/smoothing/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep100/finetune/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep100/certify_sigma0.25_test",
        # "amlt/smoothing/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep100/finetune/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep100/certify_sigma0.5_test",
        # "amlt/smoothing/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep100/finetune/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep100/certify_sigma1.0_test",

         # IN32 pretrain with equal probability (0,0.25,0.5,1.0), CF10 Finetune on clean images lr001e100
        # "amlt/smoothing/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep1/finetune/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep1/certify_sigma0.25_test",
        # "amlt/smoothing/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep1/finetune/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep1/certify_sigma0.5_test",
        # "amlt/smoothing/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep1/finetune/nep_r152gn1_in32n0n025n05n1bs128_lr01e200_ft_lr001ep1/certify_sigma1.0_test",

        "certify_sigma0.25_test",
        "certify_sigma0.5_test",
        "certify_sigma1.0_test",
       
    ]
    
    # print(args.path)
    name = os.path.basename(os.path.normpath(args.path))
    # print(name)

    plot_certified_accuracy(
        "../amlt/smoothing/analysis/plots/" + name, "CIFAR10", 3.0, [
            Line(ApproximateAccuracy(os.path.join(args.path, ctf_file)), "$\sigma = $ "+ctf_file.split('/')[-1].split('_')[-2].replace('sigma', '')) for ctf_file in ctf_files
        ])
    
    # plot_certified_accuracy(
    #     "../amlt/smoothing/analysis/plots/vary_noise_cifar10_4.0", "CIFAR-10, vary $\sigma$", 4.0, [
    #         Line(ApproximateAccuracy("../amlt/smoothing/data/predict/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
    #         Line(ApproximateAccuracy("../amlt/smoothing/data/predict/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("../amlt/smoothing/data/predict/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("../amlt/smoothing/data/predict/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
    #     ])
    # latex_table_certified_accuracy(
    #     "analysis/latex/vary_noise_cifar10", 0.25, 1.5, 0.25, [
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
    #     ])
    # markdown_table_certified_accuracy(
    #     "analysis/markdown/vary_noise_cifar10", 0.25, 1.5, 0.25, [
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "&sigma; = 0.12"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "&sigma; = 0.25"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "&sigma; = 0.50"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "&sigma; = 1.00"),
    #     ])
    # latex_table_certified_accuracy(
    #     "analysis/latex/vary_noise_imagenet", 0.5, 3.0, 0.5, [
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
    #     ])
    # markdown_table_certified_accuracy(
    #     "analysis/markdown/vary_noise_imagenet", 0.5, 3.0, 0.5, [
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "&sigma; = 0.25"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "&sigma; = 0.50"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "&sigma; = 1.00"),
    #     ])
    # plot_certified_accuracy(
    #     "analysis/plots/vary_noise_cifar10", "CIFAR-10, vary $\sigma$", 1.5, [
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.12/test/sigma_0.12"), "$\sigma = 0.12$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
    #     ])
    # plot_certified_accuracy(
    #     "analysis/plots/vary_train_noise_cifar_050", "CIFAR-10, vary train noise, $\sigma=0.5$", 1.5, [
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.25/test/sigma_0.50"), "train $\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_0.50/test/sigma_0.50"), "train $\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/cifar10/resnet110/noise_1.00/test/sigma_0.50"), "train $\sigma = 1.00$"),
    #     ])
    # plot_certified_accuracy(
    #     "analysis/plots/vary_train_noise_imagenet_050", "ImageNet, vary train noise, $\sigma=0.5$", 1.5, [
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.50"), "train $\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "train $\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_0.50"), "train $\sigma = 1.00$"),
    #     ])
    # plot_certified_accuracy(
    #     "analysis/plots/vary_noise_imagenet", "ImageNet, vary $\sigma$", 4, [
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.25/test/sigma_0.25"), "$\sigma = 0.25$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "$\sigma = 0.50$"),
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_1.00/test/sigma_1.00"), "$\sigma = 1.00$"),
    #     ])
    # plot_certified_accuracy(
    #     "analysis/plots/high_prob", "Approximate vs. High-Probability", 2.0, [
    #         Line(ApproximateAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50"), "Approximate"),
    #         Line(HighProbAccuracy("data/certify/imagenet/resnet50/noise_0.50/test/sigma_0.50", 0.001, 0.001), "High-Prob"),
    #     ])
