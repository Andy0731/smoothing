import os
import numpy as np
import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math

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
        line.legend = line.legend + '_acc{:.1f}'.format(accuracy * 100)
        plt.plot(radii * line.scale_x, robust_acc, linestyle='solid', label=line.legend + '_ra')
        # plt.plot(radii * line.scale_x, robust_acc, color=colors[idx % 7], linestyle='solid', label=line.legend)
        # plt.plot(radii * line.scale_x, robustness, color=colors[idx % 7], linestyle='dashed', label=line.legend + '_ro')
        df_list.append(spe_robust_acc_np)
        exp_list.append(line.legend + '_ra')
    
    exps = pd.DataFrame(np.array(df_list) * 100.0, index=exp_list, columns=sv)
    print(exps)
    exps.to_csv(outfile + '_ra.csv')
        
    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified accuracy", fontsize=16)
    # plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    # plt.legend([method.legend for method in lines], fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(outfile + "_ra.pdf")
    plt.tight_layout()
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + "_ra.png", dpi=300)
    plt.close()

    plt.figure()
    # colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    # assert len(lines) < 8, 'plot at most 7*2 curves!'
    df_list = []
    exp_list = []    
    for idx, line in enumerate(lines):
        robust_acc, robustness, accuracy, sv, spe_robust_acc_np, spe_robust_np = line.quantity.at_radii(radii)
        # line.legend = line.legend + '_acc{:.1f}'.format(accuracy * 100)
        # plt.plot(radii * line.scale_x, robust_acc, color=colors[idx % 7], linestyle='solid', label=line.legend)
        # plt.plot(radii * line.scale_x, robustness, color=colors[idx % 7], linestyle='dashed', label=line.legend + '_ro')
        plt.plot(radii * line.scale_x, robustness, linestyle='dashed', label=line.legend + '_r')
        df_list.append(spe_robust_np)
        exp_list.append(line.legend + '_r')

    exps = pd.DataFrame(np.array(df_list) * 100.0, index=exp_list, columns=sv)
    print(exps)
    exps.to_csv(outfile + '_r.csv')
        
    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified accuracy", fontsize=16)
    # plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    # plt.legend([method.legend for method in lines], fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(outfile + "_r.pdf")
    plt.tight_layout()
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + "_r.png", dpi=300)
    plt.close()    


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

        # "amlt/smoothing/r110_n025_coslr01_bs16_e200_cert_train/resnet110/n025_coslr01_bs16_e200_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs16_e400_cert_train/resnet110/n025_coslr01_bs16_e400_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs16_e800_cert_train/resnet110/n025_coslr01_bs16_e800_cert_train/certify_sigma0.25",

        # train from scratch on CIFAR10, resnet110 n025 coslr01 bs128, vary ep, train
        # "amlt/smoothing/r110_n025_coslr01_bs128_e200_cert_train/resnet110/n025_coslr01_bs128_e200_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e200_damix/resnet110/n025_coslr01_bs128_e200_damix/certify_sigma0.25_train",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e200_dav3/resnet110/n025_coslr01_bs128_e200_dav3/certify_sigma0.25_train",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e400_cert_train/resnet110/n025_coslr01_bs128_e400_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e800_cert_train/resnet110/n025_coslr01_bs128_e800_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e1600/resnet110/n025_coslr01_bs128_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e3200/resnet110/n025_coslr01_bs128_e3200/certify_sigma0.25_train",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e6400_m2/resnet110/n025_coslr01_bs128_e6400/certify_sigma0.25_train",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e12800_m2/resnet110/n025_coslr01_bs128_e12800/certify_sigma0.25_train",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e25600_m4/resnet110/n025_coslr01_bs128_e25600/certify_sigma0.25_train",

        # train from scratch on CIFAR10, resnet110 n025 coslr01 bs128, vary ep, test
        # "amlt/smoothing/r110_n025_coslr01_bs128_e200/resnet110/n025_coslr01_bs128_e200/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e200_damix/resnet110/n025_coslr01_bs128_e200_damix/certify_sigma0.25_test",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e200_dav3/resnet110/n025_coslr01_bs128_e200_dav3/certify_sigma0.25_test",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e400/resnet110/n025_coslr01_bs128_e400/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e800/resnet110/n025_coslr01_bs128_e800/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e1600/resnet110/n025_coslr01_bs128_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e1600_damix/resnet110/n025_coslr01_bs128_e1600_damix/certify_sigma0.25_test",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e1600_dav3/resnet110/n025_coslr01_bs128_e1600_dav3/certify_sigma0.25_test",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e3200/resnet110/n025_coslr01_bs128_e3200/certify_sigma0.25_test",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e6400_m2/resnet110/n025_coslr01_bs128_e6400/certify_sigma0.25_test",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e12800_m2/resnet110/n025_coslr01_bs128_e12800/certify_sigma0.25_test",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e25600_m4/resnet110/n025_coslr01_bs128_e25600/certify_sigma0.25_test",

        # "amlt/smoothing/r110_n025_coslr01_bs128_e1600_damix/resnet110/n025_coslr01_bs128_e1600_damix/certify_sigma0.25_train",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e1600_dav3/resnet110/n025_coslr01_bs128_e1600_dav3/certify_sigma0.25_train",        

        # "amlt/smoothing/r110_n025_lr01_bs16_e200_cert_train/resnet110/n025_lr01_bs16_e200_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_lr01_bs16_e400_cert_train/resnet110/n025_lr01_bs16_e400_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_lr01_bs16_e800_cert_train/resnet110/n025_lr01_bs16_e800_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_lr01_bs128_e200_cert_train/resnet110/n025_lr01_bs128_e200_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_lr01_bs128_e400_cert_train/resnet110/n025_lr01_bs128_e400_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_lr01_bs128_e800_cert_train/resnet110/n025_lr01_bs128_e800_cert_train/certify_sigma0.25",

        # "amlt/smoothing/consistency_paper/r110_gaussian_n025",

        # "amlt/smoothing/r152_n025_coslr01_bs16_e100/resnet152/n025_coslr01_bs16_e100/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_coslr01_bs16_e200/resnet152/n025_coslr01_bs16_e200/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_coslr01_bs16_e400/resnet152/n025_coslr01_bs16_e400/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_coslr01_bs16_e800/resnet152/n025_coslr01_bs16_e800/certify_sigma0.25",

        # "amlt/smoothing/r152_n025_coslr01_bs128_e100/resnet152/n025_coslr01_bs128_e100/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_coslr01_bs128_e200/resnet152/n025_coslr01_bs128_e200/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_coslr01_bs128_e400/resnet152/n025_coslr01_bs128_e400/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_coslr01_bs128_e800/resnet152/n025_coslr01_bs128_e800/certify_sigma0.25",

        # "amlt/smoothing/r152_n025_g4_lr01_sbn_bs16/resnet152/n025_lr01_bs16/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_g4_lr01_sbn_bs128/resnet152/n025_lr01_bs128/certify_sigma0.25",

        # "amlt/smoothing/r152_n025_lr01_bs16_e100/resnet152/n025_lr01_bs16_e100/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_lr01_bs16_e200/resnet152/n025_lr01_bs16_e200/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_lr01_bs16_e400/resnet152/n025_lr01_bs16_e400/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_lr01_bs16_e800/resnet152/n025_lr01_bs16_e800/certify_sigma0.25",

        # "amlt/smoothing/r110_n025_coslr01_bs128_e1600/resnet110/n025_coslr01_bs128_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e12800_m2/resnet110/n025_coslr01_bs128_e12800/certify_sigma0.25_test",

        # ******************************
        # train from scratch on CIFAR10, normal_resnet152, n025 coslr01 bs128, vary ep, train 
        # "amlt/smoothing/rep_r152_n025_coslr01_bs128_e100/resnet152/rep_n025_coslr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/rep_n025_coslr01_bs128_e800/resnet152/rep_n025_coslr01_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/r152_n025_coslr01_bs128_e1600_m2/resnet152/n025_coslr01_bs128_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/r152_n025_coslr01_bs128_e12800_m8/resnet152/n025_coslr01_bs128_e12800/certify_sigma0.25_train",

        # train from scratch on CIFAR10, normal_resnet152, n025 coslr01 bs128, vary ep, test
        # "amlt/smoothing/rep_r152_n025_coslr01_bs128_e100/resnet152/rep_n025_coslr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/rep_n025_coslr01_bs128_e800/resnet152/rep_n025_coslr01_bs128_e800/certify_sigma0.25_test",
        # # "amlt/smoothing/r152_n025_lr01_bs128_e100/resnet152/n025_lr01_bs128_e100/certify_sigma0.25",
        # # "amlt/smoothing/r152_n025_lr01_bs128_e200/resnet152/n025_lr01_bs128_e200/certify_sigma0.25",
        # # "amlt/smoothing/r152_n025_lr01_bs128_e400/resnet152/n025_lr01_bs128_e400/certify_sigma0.25",
        # # "amlt/smoothing/r152_n025_lr01_bs128_e800/resnet152/n025_lr01_bs128_e800/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_coslr01_bs128_e1600_m2/resnet152/n025_coslr01_bs128_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/r152_n025_coslr01_bs128_e12800_m8/resnet152/n025_coslr01_bs128_e12800/certify_sigma0.25_test",

        # "amlt/smoothing/r110_n025_lr01_bs16_e200/resnet110/n025_lr01_bs16_e200/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_lr01_bs16_e400/resnet110/n025_lr01_bs16_e400/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_lr01_bs16_e800/resnet110/n025_lr01_bs16_e800/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_lr01_bs128_e200/resnet110/n025_lr01_bs128_e200/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_lr01_bs128_e400/resnet110/n025_lr01_bs128_e400/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_lr01_bs128_e800/resnet110/n025_lr01_bs128_e800/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs16_e200/resnet110/n025_coslr01_bs16_e200/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs16_e400/resnet110/n025_coslr01_bs16_e400/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs16_e800/resnet110/n025_coslr01_bs16_e800/certify_sigma0.25",

        # "amlt/smoothing/r110_n025_g4_lr01_sbn_bs16_wu/resnet110/n025_lr01_bs16_wu/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_g4_lr01_sbn_bs128_wu/resnet110/n025_lr01_bs128_wu/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e200/resnet110/n025_coslr01_bs128_e200/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs128_e200_cert_train/resnet110/n025_coslr01_bs128_e200_cert_train/certify_sigma0.25",

        # "amlt/smoothing/r50_n025_g4_lr01_sbn_bs16_m4/resnet50/n025_lr01_bs16_m4/certify_sigma0.25",
        # "amlt/smoothing/r50_n025_g4_lr01_sbn_bs16_m4_wu/resnet50/n025_lr01_bs16_m4_wu/certify_sigma0.25",
        # "amlt/smoothing/r50_n025_g4_lr1_sbn_bs128_m4/resnet50/n025_lr1_bs128_m4/certify_sigma0.25",
        # "amlt/smoothing/r50_n025_g4_lr01_sbn_bs128_m4/resnet50/n025_lr01_bs128_m4/certify_sigma0.25",
        # "amlt/smoothing/r50_n025_g4_lr001_sbn_bs128_m4/resnet50/n025_lr001_bs128_m4/certify_sigma0.25",
        # "amlt/smoothing/r50_n025_g4_lr0001_sbn_bs128_m4/resnet50/n025_lr0001_bs128_m4/certify_sigma0.25",
        # "amlt/smoothing/r50_n025_g4_lr01_sbn_bs128_m4_wu/resnet50/n025_lr01_bs128_m4_wu/certify_sigma0.25",

        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs16/resnet110/n0_lr01_bs16/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs32/resnet110/n0_lr01_bs32/ceApproximateAccuracyrtify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs64/resnet110/n0_lr01_bs64/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs128/resnet110/n0_lr01_bs128/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs256/resnet110/n0_lr01_bs256/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs512/resnet110/n0_lr01_bs512/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs1024/resnet110/n0_lr01_bs1024/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs16_wu/resnet110/n0_lr01_bs16_wu/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs128_wu/resnet110/n0_lr01_bs128_wu/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs1024_wu/resnet110/n0_lr01_bs1024_wu/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs1024_m2/resnet110/n0_lr01_bs1024_m2/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs1024_m4/resnet110/n0_lr01_bs1024_m4/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs1024_m8/resnet110/n0_lr01_bs1024_m8/certify_sigma0.25",
        # "amlt/smoothing/r110_n0_g4_lr01_sbn_bs1024_m16/resnet110/n0_lr01_bs1024_m16/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_g4_lr01_sbn_bs1024/resnet110/n025_lr01_bs1024/certify_sigma0.25",

        # imagenet32
        # r110 n0 bs256 e100, vary lr, train
        # "amlt/smoothing/imgn32_r110_n0_coslr001_bs256_e100/imagenet32/r110_n0_coslr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr01_bs256_e100/imagenet32/r110_n0_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e100/imagenet32/r110_n0_coslr1_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr10_bs256_e100/imagenet32/r110_n0_coslr10_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr100_bs256_e100/imagenet32/r110_n0_coslr100_e100/certify_sigma0.25_train",

        # # r110 n0 bs256 e100, vary lr, test
        # "amlt/smoothing/imgn32_r110_n0_coslr001_bs256_e100/imagenet32/r110_n0_coslr001_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n0_coslr01_bs256_e100/imagenet32/r110_n0_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e100/imagenet32/r110_n0_coslr1_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n0_coslr10_bs256_e100/imagenet32/r110_n0_coslr10_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n0_coslr100_bs256_e100/imagenet32/r110_n0_coslr100_e100/certify_sigma0.25_test",

        # # r110 n0 bs256 coslr1, coslr1, vary ep, train
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e100/imagenet32/r110_n0_coslr1_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e200/imagenet32/r110_n0_coslr1_e200/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e400/imagenet32/r110_n0_coslr1_e400/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e800/imagenet32/r110_n0_coslr1_e800/certify_sigma0.25_train",    

        # # r110 n0 bs256 coslr1, coslr1, vary ep, test
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e100/imagenet32/r110_n0_coslr1_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e200/imagenet32/r110_n0_coslr1_e200/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e400/imagenet32/r110_n0_coslr1_e400/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e800/imagenet32/r110_n0_coslr1_e800/certify_sigma0.25_test",  

        # # r110 n025 bs256 e100 train
        # "amlt/smoothing/imgn32_r110_n025_coslr001_bs256_e100/imagenet32/r110_n025_coslr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n025_coslr01_bs256_e100/imagenet32/r110_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n025_coslr1_bs256_e100/imagenet32/r110_n025_coslr1_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n025_coslr10_bs256_e100/imagenet32/r110_n025_coslr10_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n025_coslr100_bs256_e100/imagenet32/r110_n025_coslr100_e100/certify_sigma0.25_train"  

        # # r110 n025 bs256 e100 test
        # "amlt/smoothing/imgn32_r110_n025_coslr001_bs256_e100/imagenet32/r110_n025_coslr001_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n025_coslr01_bs256_e100/imagenet32/r110_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n025_coslr1_bs256_e100/imagenet32/r110_n025_coslr1_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n025_coslr10_bs256_e100/imagenet32/r110_n025_coslr10_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n025_coslr100_bs256_e100/imagenet32/r110_n025_coslr100_e100/certify_sigma0.25_test"

        # r110, pretrain on imagenet32 with noise n025 bs256 coslr1 train
        # "amlt/smoothing/imgn32_r110_n025_coslr1_bs256_e100/imagenet32/r110_n025_coslr1_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n025_coslr1_bs256_e200/imagenet32/r110_n025_coslr1_e200/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n025_coslr1_bs256_e400/imagenet32/r110_n025_coslr1_e400/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n025_coslr1_bs256_e800/imagenet32/r110_n025_coslr1_e800/certify_sigma0.25_train",    

        # # r110, pretrain on imagenet32 with noise n025 bs256 coslr1 test
        # "amlt/smoothing/imgn32_r110_n025_coslr1_bs256_e100/imagenet32/r110_n025_coslr1_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n025_coslr1_bs256_e200/imagenet32/r110_n025_coslr1_e200/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n025_coslr1_bs256_e400/imagenet32/r110_n025_coslr1_e400/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n025_coslr1_bs256_e800/imagenet32/r110_n025_coslr1_e800/certify_sigma0.25_test",

        # r152, pretrain on imagenet32 with noise n025 bs256 coslr1 train
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e100_m8/imagenet32/r152_n025_coslr1_e100/certify_sigma0.25_train",

        # r152, pretrain on imagenet32 with noise n025 bs256 coslr1 test
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e100_m8/imagenet32/r152_n025_coslr1_e100/certify_sigma0.25_test",

        # # r110, pretrain on imagenet32 without noise, finetune on cifar10, cosin lr, train
        # "amlt/smoothing/imgn32n0_coslr01_bs128_n025_e10/finetune/imgn32n0_coslr01_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n0_coslr001_bs128_n025_e10/finetune/imgn32n0_coslr001_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n0_coslr0001_bs128_n025_e10/finetune/imgn32n0_coslr0001_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n0_coslr01_bs128_n025_e100/finetune/imgn32n0_coslr01_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n0_coslr001_bs128_n025_e100/finetune/imgn32n0_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n0_coslr0001_bs128_n025_e100/finetune/imgn32n0_coslr0001_bs128_n025_e100/certify_sigma0.25_train",

        # # r110, pretrain on imagenet32 without noise, finetune on cifar10, cosin lr, test
        # "amlt/smoothing/imgn32n0_coslr01_bs128_n025_e10/finetune/imgn32n0_coslr01_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n0_coslr001_bs128_n025_e10/finetune/imgn32n0_coslr001_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n0_coslr0001_bs128_n025_e10/finetune/imgn32n0_coslr0001_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n0_coslr01_bs128_n025_e100/finetune/imgn32n0_coslr01_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n0_coslr001_bs128_n025_e100/finetune/imgn32n0_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n0_coslr0001_bs128_n025_e100/finetune/imgn32n0_coslr0001_bs128_n025_e100/certify_sigma0.25_test",

        # # r110, pretrain on imagenet32 without noise, finetune on cifar10, constant lr, train
        # "amlt/smoothing/imgn32n0_lr01_bs128_n025_e10/finetune/imgn32n0_lr01_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n0_lr001_bs128_n025_e10/finetune/imgn32n0_lr001_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n0_lr0001_bs128_n025_e10/finetune/imgn32n0_lr0001_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n0_lr01_bs128_n025_e100/finetune/imgn32n0_lr01_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n0_lr001_bs128_n025_e100/finetune/imgn32n0_lr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n0_lr0001_bs128_n025_e100/finetune/imgn32n0_lr0001_bs128_n025_e100/certify_sigma0.25_train",

        # # r110, pretrain on imagenet32 without noise, finetune on cifar10, constant lr, test
        # "amlt/smoothing/imgn32n0_lr01_bs128_n025_e10/finetune/imgn32n0_lr01_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n0_lr001_bs128_n025_e10/finetune/imgn32n0_lr001_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n0_lr0001_bs128_n025_e10/finetune/imgn32n0_lr0001_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n0_lr01_bs128_n025_e100/finetune/imgn32n0_lr01_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n0_lr001_bs128_n025_e100/finetune/imgn32n0_lr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n0_lr0001_bs128_n025_e100/finetune/imgn32n0_lr0001_bs128_n025_e100/certify_sigma0.25_test",

        # # r110, pretrain on imagenet32 with noise_sd 0.25, finetune  on cifar10, consin lr, train
        # "amlt/smoothing/imgn32n025_coslr01_bs128_n025_e10/finetune/imgn32n025_coslr01_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e10/finetune/imgn32n025_coslr001_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr0001_bs128_n025_e10/finetune/imgn32n025_coslr0001_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr01_bs128_n025_e100/finetune/imgn32n025_coslr01_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e100/finetune/imgn32n025_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr0001_bs128_n025_e100/finetune/imgn32n025_coslr0001_bs128_n025_e100/certify_sigma0.25_train",

        # # r110, pretrain on imagenet32 with noise_sd 0.25, finetune  on cifar10, consin lr, test
        # "amlt/smoothing/imgn32n025_coslr01_bs128_n025_e10/finetune/imgn32n025_coslr01_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e10/finetune/imgn32n025_coslr001_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr0001_bs128_n025_e10/finetune/imgn32n025_coslr0001_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr01_bs128_n025_e100/finetune/imgn32n025_coslr01_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e100/finetune/imgn32n025_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr0001_bs128_n025_e100/finetune/imgn32n025_coslr0001_bs128_n025_e100/certify_sigma0.25_test",

        # r110, pretrain on imagenet32 with noise_sd 0.25, finetune  on cifar10, consin lr 0.01, train
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e5/finetune/imgn32n025_coslr001_bs128_n025_e5/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e10/finetune/imgn32n025_coslr001_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e20/finetune/imgn32n025_coslr001_bs128_n025_e20/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e40/finetune/imgn32n025_coslr001_bs128_n025_e40/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e60/finetune/imgn32n025_coslr001_bs128_n025_e60/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e80/finetune/imgn32n025_coslr001_bs128_n025_e80/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e100/finetune/imgn32n025_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e200/finetune/imgn32n025_coslr001_bs128_n025_e200/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e400/finetune/imgn32n025_coslr001_bs128_n025_e400/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e800/finetune/imgn32n025_coslr001_bs128_n025_e800/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e1600/finetune/imgn32n025_coslr001_bs128_n025_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e3200/finetune/imgn32n025_coslr001_bs128_n025_e3200/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e6400/finetune/imgn32n025_coslr001_bs128_n025_e6400/certify_sigma0.25_train",

        # r110, pretrain on imagenet32 with noise_sd 0.25, finetune  on cifar10, consin lr 0.01, test
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e5/finetune/imgn32n025_coslr001_bs128_n025_e5/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e10/finetune/imgn32n025_coslr001_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e20/finetune/imgn32n025_coslr001_bs128_n025_e20/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e40/finetune/imgn32n025_coslr001_bs128_n025_e40/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e60/finetune/imgn32n025_coslr001_bs128_n025_e60/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e80/finetune/imgn32n025_coslr001_bs128_n025_e80/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e100/finetune/imgn32n025_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e200/finetune/imgn32n025_coslr001_bs128_n025_e200/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e400/finetune/imgn32n025_coslr001_bs128_n025_e400/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e800/finetune/imgn32n025_coslr001_bs128_n025_e800/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e1600/finetune/imgn32n025_coslr001_bs128_n025_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e3200/finetune/imgn32n025_coslr001_bs128_n025_e3200/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_coslr001_bs128_n025_e6400/finetune/imgn32n025_coslr001_bs128_n025_e6400/certify_sigma0.25_test",


        # # r110, pretrain on imagenet32 with noise_se 0.25, finetune on cifar10, constant lr, train
        # "amlt/smoothing/imgn32n025_lr01_bs128_n025_e10/finetune/imgn32n025_lr01_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_lr001_bs128_n025_e10/finetune/imgn32n025_lr001_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_lr0001_bs128_n025_e10/finetune/imgn32n025_lr0001_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_lr01_bs128_n025_e100/finetune/imgn32n025_lr01_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_lr001_bs128_n025_e100/finetune/imgn32n025_lr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025_lr0001_bs128_n025_e100/finetune/imgn32n025_lr0001_bs128_n025_e100/certify_sigma0.25_train",

        # # r110, pretrain on imagenet32 with noise_se 0.25, finetune on cifar10, constant lr, test
        # "amlt/smoothing/imgn32n025_lr01_bs128_n025_e10/finetune/imgn32n025_lr01_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_lr001_bs128_n025_e10/finetune/imgn32n025_lr001_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_lr0001_bs128_n025_e10/finetune/imgn32n025_lr0001_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_lr01_bs128_n025_e100/finetune/imgn32n025_lr01_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_lr001_bs128_n025_e100/finetune/imgn32n025_lr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025_lr0001_bs128_n025_e100/finetune/imgn32n025_lr0001_bs128_n025_e100/certify_sigma0.25_test",

        # # r110, pretrain on imagenet32 with noise_sd 0.25 vary epochs, finetune on cifar10 100 epochs cosin lr 0.01, train
        # "amlt/smoothing/imgn32n025e100_coslr001_bs128_n025_e100/finetune/imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025e200_coslr001_bs128_n025_e100/finetune/imgn32n025e200_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025e400_coslr001_bs128_n025_e100/finetune/imgn32n025e400_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025e800_coslr001_bs128_n025_e100/finetune/imgn32n025e800_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # # r110, pretrain on imagenet32 with noise_sd 0.25 vary epochs, finetune on cifar10 100 epochs cosin lr 0.01, test
        # "amlt/smoothing/imgn32n025e100_coslr001_bs128_n025_e100/finetune/imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025e200_coslr001_bs128_n025_e100/finetune/imgn32n025e200_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025e400_coslr001_bs128_n025_e100/finetune/imgn32n025e400_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025e800_coslr001_bs128_n025_e100/finetune/imgn32n025e800_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # r110, pretrain on imagenet32 with noise_sd 0.25 vary epochs, finetune on cifar10 40 epochs cosin lr 0.01, train
        # "amlt/smoothing/imgn32n025e100_coslr001_bs128_n025_e40/finetune/imgn32n025e100_coslr001_bs128_n025_e40/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025e200_coslr001_bs128_n025_e40/finetune/imgn32n025e200_coslr001_bs128_n025_e40/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025e400_coslr001_bs128_n025_e40/finetune/imgn32n025e400_coslr001_bs128_n025_e40/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32n025e800_coslr001_bs128_n025_e40/finetune/imgn32n025e800_coslr001_bs128_n025_e40/certify_sigma0.25_train",

        # # r110, pretrain on imagenet32 with noise_sd 0.25 vary epochs, finetune on cifar10 40 epochs cosin lr 0.01, test
        # "amlt/smoothing/imgn32n025e100_coslr001_bs128_n025_e40/finetune/imgn32n025e100_coslr001_bs128_n025_e40/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025e200_coslr001_bs128_n025_e40/finetune/imgn32n025e200_coslr001_bs128_n025_e40/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025e400_coslr001_bs128_n025_e40/finetune/imgn32n025e400_coslr001_bs128_n025_e40/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32n025e800_coslr001_bs128_n025_e40/finetune/imgn32n025e800_coslr001_bs128_n025_e40/certify_sigma0.25_test",

        # # r152 pretrain on imagenet32 with noise_sd 0.25, vary epochs, train
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e100_m8/imagenet32/r152_n025_coslr1_e100/certify_sigma0.25_train",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr01_e100_m8/imagenet32/r152_n025_coslr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e200/imagenet32/r152_n025_coslr1_e200/certify_sigma0.25_train",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e400_m16/imagenet32/r152_n025_coslr1_e400/certify_sigma0.25_train",

        # # r152 pretrain on imagenet32 with noise_sd 0.25, vary epochs, test
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e100_m8/imagenet32/r152_n025_coslr1_e100/certify_sigma0.25_test",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr01_e100_m8/imagenet32/r152_n025_coslr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e200/imagenet32/r152_n025_coslr1_e200/certify_sigma0.25_test",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e400_m16/imagenet32/r152_n025_coslr1_e400/certify_sigma0.25_test",

        # # r152 pretrain on imagenet32 with noise_sd 0.25 e100, finetune on cifar10, vary lr, vary epochs ; train
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr01_bs128_n025_e10/finetune/r152_imgn32n025e100_coslr01_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e10/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr0001_bs128_n025_e10/finetune/r152_imgn32n025e100_coslr0001_bs128_n025_e10/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr01_bs128_n025_e40/finetune/r152_imgn32n025e100_coslr01_bs128_n025_e40/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e40/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e40/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr0001_bs128_n025_e40/finetune/r152_imgn32n025e100_coslr0001_bs128_n025_e40/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr01_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr01_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr0001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr0001_bs128_n025_e100/certify_sigma0.25_train",

        # r152 pretrain on imagenet32 with noise_sd 0.25 e100, finetune on cifar10, vary lr, vary epochs ; test
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr01_bs128_n025_e10/finetune/r152_imgn32n025e100_coslr01_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e10/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr0001_bs128_n025_e10/finetune/r152_imgn32n025e100_coslr0001_bs128_n025_e10/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr01_bs128_n025_e40/finetune/r152_imgn32n025e100_coslr01_bs128_n025_e40/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e40/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e40/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr0001_bs128_n025_e40/finetune/r152_imgn32n025e100_coslr0001_bs128_n025_e40/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr01_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr01_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr0001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr0001_bs128_n025_e100/certify_sigma0.25_test",

        # *******************************
        # # r152 pretrain on imagenet32 with noise_sd 0.25, vary epochs, finetune on cifar10 coslr001 bs128 n025 e100, train
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e200_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e200_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e400_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e400_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # # r152 pretrain on imagenet32 with noise_sd 0.25, vary epochs, finetune on cifar10 coslr001 bs128 n025 e100, test
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e200_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e200_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e400_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e400_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # # r152 pretrain on imagenet32 without noise e100, finetune on cifar10 coslr001 bs128 n025 e100, train
        # "amlt/smoothing/ft_r152_imgn32n0e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n0e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # # r152 pretrain on imagenet32 without noise e100, finetune on cifar10 coslr001 bs128 n025 e100, train
        # "amlt/smoothing/ft_r152_imgn32n0e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n0e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # # r152 pretrain on iamgenet32 with n025 vary ep, finetune on cifar10 lr001 bs128 n025, vary ep, train
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e800/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e800/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e1600/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e400_coslr001_bs128_n025_e800/finetune/r152_imgn32n025e400_coslr001_bs128_n025_e800/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e400_coslr001_bs128_n025_e1600/finetune/r152_imgn32n025e400_coslr001_bs128_n025_e1600/certify_sigma0.25_train",

        # r152 pretrain on iamgenet32 with n025 vary ep, finetune on cifar10 lr001 bs128 n025, vary ep, test
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e800/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e800/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e1600/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e400_coslr001_bs128_n025_e800/finetune/r152_imgn32n025e400_coslr001_bs128_n025_e800/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e400_coslr001_bs128_n025_e1600/finetune/r152_imgn32n025e400_coslr001_bs128_n025_e1600/certify_sigma0.25_test",

        # # # r152 pretrain on imagenet32 with noise_sd 0.25, vary epochs, finetune on cifar10 coslr001 bs128 n025 e40, train
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e40/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e40/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e200_coslr001_bs128_n025_e40/finetune/r152_imgn32n025e200_coslr001_bs128_n025_e40/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e400_coslr001_bs128_n025_e40/finetune/r152_imgn32n025e400_coslr001_bs128_n025_e40/certify_sigma0.25_train",

        # # r152 pretrain on imagenet32 with noise_sd 0.25, vary epochs, finetune on cifar10 coslr001 bs128 n025 e40, test
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e40/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e40/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e200_coslr001_bs128_n025_e40/finetune/r152_imgn32n025e200_coslr001_bs128_n025_e40/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e400_coslr001_bs128_n025_e40/finetune/r152_imgn32n025e400_coslr001_bs128_n025_e40/certify_sigma0.25_test",

        # # r152 pretrain on ti500k with n025, vary lr, vary epoch
        # "amlt/smoothing/pt_ti500k_r152_n025_coslr01_e100/ti500k/r152_n025_coslr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/pt_ti500k_r152_n025_coslr01_e200/ti500k/r152_n025_coslr01_e200/certify_sigma0.25_train",
        # "amlt/smoothing/pt_ti500k_r152_n025_coslr01_e400/ti500k/r152_n025_coslr01_e400/certify_sigma0.25_train",

        # "amlt/smoothing/pt_ti500k_r152_n025_coslr001_e100/ti500k/r152_n025_coslr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/pt_ti500k_r152_n025_coslr001_e200/ti500k/r152_n025_coslr001_e200/certify_sigma0.25_train",
        # "amlt/smoothing/pt_ti500k_r152_n025_coslr001_e400/ti500k/r152_n025_coslr001_e400/certify_sigma0.25_train",
        
        # "amlt/smoothing/pt_ti500k_r152_n025_coslr0001_e100/ti500k/r152_n025_coslr0001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/pt_ti500k_r152_n025_coslr0001_e200/ti500k/r152_n025_coslr0001_e200/certify_sigma0.25_train",

        # # r152 pretrain on ti500k with n025, lr01 e100, finetune on cifar10 vary epoch, vary lr, train
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr01_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr01_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr0001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr0001_bs128_n025_e100/certify_sigma0.25_train",

        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr01_bs128_n025_e40_P100/finetune/r152_ti500kn025lr01e100_coslr01_bs128_n025_e40/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr001_bs128_n025_e40_P100/finetune/r152_ti500kn025lr01e100_coslr001_bs128_n025_e40/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr0001_bs128_n025_e40_P100/finetune/r152_ti500kn025lr01e100_coslr0001_bs128_n025_e40/certify_sigma0.25_train",

        #  # r152 pretrain on ti500k with n025, lr01 e100, finetune on cifar10 vary epoch, vary lr, test
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr01_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr01_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr0001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr0001_bs128_n025_e100/certify_sigma0.25_test",

        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr01_bs128_n025_e40_P100/finetune/r152_ti500kn025lr01e100_coslr01_bs128_n025_e40/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr001_bs128_n025_e40_P100/finetune/r152_ti500kn025lr01e100_coslr001_bs128_n025_e40/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr0001_bs128_n025_e40_P100/finetune/r152_ti500kn025lr01e100_coslr0001_bs128_n025_e40/certify_sigma0.25_test",

        # # r152 pretrain on ti500k with n025, vary lr, vary ep, finetune on cifar10 lr001 bs128 n025 e100, train
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e200_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e200_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e400_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e400_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e800_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e800_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # "amlt/smoothing/ft_r152_ti500kn025lr001e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr001e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_ti500kn025lr001e200_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr001e200_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_ti500kn025lr001e400_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr001e400_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # "amlt/smoothing/ft_r152_ti500kn025lr0001e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr0001e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_ti500kn025lr0001e200_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr0001e200_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # # r152 pretrain on ti500k with n025, vary lr, vary ep, finetune on cifar10 lr001 bs128 n025 e100, test
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e200_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e200_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e400_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e400_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e800_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e800_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # "amlt/smoothing/ft_r152_ti500kn025lr001e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr001e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr001e200_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr001e200_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr001e400_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr001e400_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # "amlt/smoothing/ft_r152_ti500kn025lr0001e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr0001e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr0001e200_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr0001e200_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # # r300 vs r152, pretrain on imagenet32
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e100_m8/imagenet32/r152_n025_coslr1_e100/certify_sigma0.25_train",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e200/imagenet32/r152_n025_coslr1_e200/certify_sigma0.25_train",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e400_m16/imagenet32/r152_n025_coslr1_e400/certify_sigma0.25_train",
        # "amlt/smoothing/pt_imgn32_r300_n025_coslr01_e100/imagenet32/r300_n025_coslr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e100_m8/imagenet32/r152_n025_coslr1_e100/certify_sigma0.25_test",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e200/imagenet32/r152_n025_coslr1_e200/certify_sigma0.25_test",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e400_m16/imagenet32/r152_n025_coslr1_e400/certify_sigma0.25_test",
        # "amlt/smoothing/pt_imgn32_r300_n025_coslr01_e100/imagenet32/r300_n025_coslr01_e100/certify_sigma0.25_test",

        # # r300 pretrain on imagenet32 with n025 lr01 100ep, finetune on cifar10 100ep, vary lr, train
        # "amlt/smoothing/ft_r300_imgn32n025e100_coslr01_bs128_n025_e100/finetune/r300_imgn32n025e100_coslr01_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r300_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r300_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r300_imgn32n025e100_coslr0001_bs128_n025_e100/finetune/r300_imgn32n025e100_coslr0001_bs128_n025_e100/certify_sigma0.25_train",

        # r300 pretrain on imagenet32 with n025 lr01 100ep, finetune on cifar10 100ep, vary lr, train
        # "amlt/smoothing/ft_r300_imgn32n025e100_coslr01_bs128_n025_e100/finetune/r300_imgn32n025e100_coslr01_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r300_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r300_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r300_imgn32n025e100_coslr0001_bs128_n025_e100/finetune/r300_imgn32n025e100_coslr0001_bs128_n025_e100/certify_sigma0.25_test",

        # # r300 vs r152, imgn32 vs ti500k, pretrain + finetune, train
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",     
        # "amlt/smoothing/ft_r300_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r300_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # r300 vs r152, imgn32 vs ti500k, pretrain + finetune, test
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",     
        # "amlt/smoothing/ft_r300_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r300_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152w2_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r152w2_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # # moco r152 pretrain on imagenet32, finetune on cifar10, test
        # "amlt/smoothing/ft_moco_r152cropn025e100_coslr001_bs128_n025_e100/finetune/moco_r152cropn025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_r1522crope100_coslr001_bs128_n025_e100/finetune/moco_r1522crope100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_r1522croppad32e100_coslr001_bs128_n025_e100/finetune/moco_r1522croppad32e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_r1522onlynoise025e100lr006_coslr001_bs128_n025_e100/finetune/moco_r1522onlynoise025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # # increasing noise
        # "amlt/smoothing/tc_r152_n025base_coslr01_bs128_e100/resnet152/tc_n025base_coslr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tc_r152_n025line50_coslr01_bs128_e100/resnet152/tc_n025line50_coslr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tc_r152_n025line100_coslr01_bs128_e100/resnet152/tc_n025line100_coslr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tc_r152_n025step1e50_coslr01_bs128_e100/resnet152/tc_n025step1e50_coslr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tc_r152_n025step5e50_coslr01_bs128_e100/resnet152/tc_n025step5e50_coslr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tc_r152_n025step5e100_coslr01_bs128_e100/resnet152/tc_n025step5e100_coslr01_bs128_e100/certify_sigma0.25_test",

        # # normal_resnet152 MOCO pretrain on imagenet32 lr06, finetune on cifar10 lr001 e100, train
        # "amlt/smoothing/ft_moco_nmrr152cropn025e100_coslr001_bs128_n025_e100/finetune/moco_nmrr152cropn025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr152cropn025e200_coslr001_bs128_n025_e100/finetune/moco_nmrr152cropn025e200_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr1521crop2noisee100_coslr001_bs128_n025_e100/finetune/moco_nmrr1521crop2noisee100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr1522crope100_coslr001_bs128_n025_e100/finetune/moco_nmrr1522crope100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr1522noise025e100_coslr001_bs128_n025_e100/finetune/moco_nmrr1522noise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr1522onlynoise025e100_coslr001_bs128_n025_e100/finetune/moco_nmrr1522onlynoise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train"

        # "amlt/smoothing/ft_moco_nmrr1522noise025e200_coslr001_bs128_n025_e100/finetune/moco_nmrr1522noise025e200_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr1522noise025e400_coslr001_bs128_n025_e100/finetune/moco_nmrr1522noise025e400_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr1522noise025e800_coslr001_bs128_n025_e100/finetune/moco_nmrr1522noise025e800_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # # # normal_resnet152 MOCO pretrain on imagenet32 lr06, finetune on cifar10 lr001 e100, test
        # "amlt/smoothing/ft_moco_nmrr152cropn025e100_coslr001_bs128_n025_e100/finetune/moco_nmrr152cropn025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr152cropn025e200_coslr001_bs128_n025_e100/finetune/moco_nmrr152cropn025e200_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr1521crop2noisee100_coslr001_bs128_n025_e100/finetune/moco_nmrr1521crop2noisee100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr1522crope100_coslr001_bs128_n025_e100/finetune/moco_nmrr1522crope100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr1522noise025e100_coslr001_bs128_n025_e100/finetune/moco_nmrr1522noise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr1522onlynoise025e100_coslr001_bs128_n025_e100/finetune/moco_nmrr1522onlynoise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test"

        # "amlt/smoothing/ft_moco_nmrr1522noise025e200_coslr001_bs128_n025_e100/finetune/moco_nmrr1522noise025e200_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr1522noise025e400_coslr001_bs128_n025_e100/finetune/moco_nmrr1522noise025e400_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr1522noise025e800_coslr001_bs128_n025_e100/finetune/moco_nmrr1522noise025e800_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # # normal_resnet152 MOCO pretrain on imagenet32 lr006, finetune on cifar10 lr001 e100, train
        # "amlt/smoothing/ft_moco_nmrr152cropn025e100lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr152cropn025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr152cropn025e200lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr152cropn025e200lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr1521crop2noisee100lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr1521crop2noisee100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr1522crope100lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr1522crope100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr1522noise025e100lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr1522noise025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_nmrr1522onlynoise025e100lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr1522onlynoise025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # # normal_resnet152 MOCO pretrain on imagenet32 lr006, finetune on cifar10 lr001 e100, test
        # "amlt/smoothing/ft_moco_nmrr152cropn025e100lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr152cropn025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr152cropn025e200lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr152cropn025e200lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr1521crop2noisee100lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr1521crop2noisee100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr1522crope100lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr1522crope100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr1522noise025e100lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr1522noise025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_nmrr1522onlynoise025e100lr006_coslr001_bs128_n025_e100/finetune/moco_nmrr1522onlynoise025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # # # normal_resnet152 MOCO pretrain on ti500k lr06, finetune on cifar10 lr001 e100, train
        # "amlt/smoothing/ft_moco_ti500k_nmrr152cropn025e100_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr152cropn025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_ti500k_nmrr152cropn025e200_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr152cropn025e200_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1521crop2noisee100_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1521crop2noisee100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522crope100_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522crope100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522noise025e100_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522noise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522onlynoise025e100_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522onlynoise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # # # normal_resnet152 MOCO pretrain on ti500k lr06, finetune on cifar10 lr001 e100, test
        # "amlt/smoothing/ft_moco_ti500k_nmrr152cropn025e100_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr152cropn025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_ti500k_nmrr152cropn025e200_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr152cropn025e200_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1521crop2noisee100_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1521crop2noisee100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522crope100_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522crope100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522noise025e100_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522noise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522onlynoise025e100_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522onlynoise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # # normal_resnet152 MOCO pretrain on ti500k lr006, finetune on cifar10 lr001 e100, train
        # "amlt/smoothing/ft_moco_ti500k_nmrr152cropn025e100lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr152cropn025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_ti500k_nmrr152cropn025e200lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr152cropn025e200lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1521crop2noisee100lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1521crop2noisee100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522crope100lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522crope100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522noise025e100lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522noise025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522onlynoise025e100lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522onlynoise025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # # normal_resnet152 MOCO pretrain on ti500k lr006, finetune on cifar10 lr001 e100, test
        # "amlt/smoothing/ft_moco_ti500k_nmrr152cropn025e100lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr152cropn025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_ti500k_nmrr152cropn025e200lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr152cropn025e200lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1521crop2noisee100lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1521crop2noisee100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522crope100lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522crope100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522noise025e100lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522noise025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_moco_ti500k_nmrr1522onlynoise025e100lr006_coslr001_bs128_n025_e100/finetune/moco_ti500k_nmrr1522onlynoise025e100lr006_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # # mixup, normal_resnet152, cifar10, train from scratch + finetune, train
        # "amlt/smoothing/mixup_n0_coslr01_bs128_e100/resnet152/mixup_n0_coslr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_n0_coslr01_bs128_e800/resnet152/mixup_n0_coslr01_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_n025_coslr01_bs128_e100/resnet152/mixup_n025_coslr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_n025_coslr01_bs128_e800/resnet152/mixup_n025_coslr01_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/mixup_r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_r152_imgn32n025e100_coslr001_bs128_n025_e800/finetune/mixup_r152_imgn32n025e100_coslr001_bs128_n025_e800/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e100_coslr001_bs128_n025_e100/finetune/mixup_moco_nmrr1522noise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e100_coslr001_bs128_n025_e800/finetune/mixup_moco_nmrr1522noise025e100_coslr001_bs128_n025_e800/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e200_coslr001_bs128_n025_e100/finetune/mixup_moco_nmrr1522noise025e200_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e200_coslr001_bs128_n025_e800/finetune/mixup_moco_nmrr1522noise025e200_coslr001_bs128_n025_e800/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e400_coslr001_bs128_n025_e100/finetune/mixup_moco_nmrr1522noise025e400_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e400_coslr001_bs128_n025_e800/finetune/mixup_moco_nmrr1522noise025e400_coslr001_bs128_n025_e800/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e800_coslr001_bs128_n025_e100/finetune/mixup_moco_nmrr1522noise025e800_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e800_coslr001_bs128_n025_e800/finetune/mixup_moco_nmrr1522noise025e800_coslr001_bs128_n025_e800/certify_sigma0.25_train",
        # "amlt/smoothing/mixup_moco_ti500k_nmrr1522noise025e100_coslr001_bs128_n025_e100/finetune/mixup_moco_ti500k_nmrr1522noise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/pt_mixup_imgn32_r152_n025_coslr01_e100/imagenet32/mixup_r152_n025_coslr01_e100/certify_sigma0.25_train",

        # mixup, normal_resnet152, cifar10, train from scratch + finetune, test
        # "amlt/smoothing/mixup_n0_coslr01_bs128_e100/resnet152/mixup_n0_coslr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_n0_coslr01_bs128_e800/resnet152/mixup_n0_coslr01_bs128_e800/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_n025_coslr01_bs128_e100/resnet152/mixup_n025_coslr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_n025_coslr01_bs128_e800/resnet152/mixup_n025_coslr01_bs128_e800/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/mixup_r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_r152_imgn32n025e100_coslr001_bs128_n025_e800/finetune/mixup_r152_imgn32n025e100_coslr001_bs128_n025_e800/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e100_coslr001_bs128_n025_e100/finetune/mixup_moco_nmrr1522noise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e100_coslr001_bs128_n025_e800/finetune/mixup_moco_nmrr1522noise025e100_coslr001_bs128_n025_e800/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e200_coslr001_bs128_n025_e100/finetune/mixup_moco_nmrr1522noise025e200_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        #  "amlt/smoothing/mixup_moco_nmrr1522noise025e200_coslr001_bs128_n025_e800/finetune/mixup_moco_nmrr1522noise025e200_coslr001_bs128_n025_e800/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e400_coslr001_bs128_n025_e100/finetune/mixup_moco_nmrr1522noise025e400_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e400_coslr001_bs128_n025_e800/finetune/mixup_moco_nmrr1522noise025e400_coslr001_bs128_n025_e800/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e800_coslr001_bs128_n025_e100/finetune/mixup_moco_nmrr1522noise025e800_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_moco_nmrr1522noise025e800_coslr001_bs128_n025_e800/finetune/mixup_moco_nmrr1522noise025e800_coslr001_bs128_n025_e800/certify_sigma0.25_test",
        # "amlt/smoothing/mixup_moco_ti500k_nmrr1522noise025e100_coslr001_bs128_n025_e100/finetune/mixup_moco_ti500k_nmrr1522noise025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/pt_mixup_imgn32_r152_n025_coslr01_e100/imagenet32/mixup_r152_n025_coslr01_e100/certify_sigma0.25_test",

        # gelu normal_resnet152, cifar10, trian from scatch with n025, train
        # "amlt/smoothing/gelu_n025_coslr01_bs128_e100/resnet152/gelu_n025_coslr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/gelu_n025_coslr01_bs128_e800/resnet152/gelu_n025_coslr01_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/gelu_n025_coslr01_bs128_e1600/resnet152/gelu_n025_coslr01_bs128_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/gelu_n025_coslr001_bs128_e100/resnet152/gelu_n025_coslr001_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/gelu_n025_coslr001_bs128_e800/resnet152/gelu_n025_coslr001_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/gelu_n025_coslr001_bs128_e1600/resnet152/gelu_n025_coslr001_bs128_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/eval_relu2gelu_n025_coslr01_bs128_e100/resnet152/eval_relu2gelu_n025_coslr01_bs128_e100/certify_sigma0.25_train",

        # gelu normal_resnet152, cifar10, trian from scatch with n025, test
        # "amlt/smoothing/gelu_n025_coslr01_bs128_e100/resnet152/gelu_n025_coslr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/gelu_n025_coslr01_bs128_e800/resnet152/gelu_n025_coslr01_bs128_e800/certify_sigma0.25_test",
        # "amlt/smoothing/gelu_n025_coslr01_bs128_e1600/resnet152/gelu_n025_coslr01_bs128_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/gelu_n025_coslr001_bs128_e100/resnet152/gelu_n025_coslr001_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/gelu_n025_coslr001_bs128_e800/resnet152/gelu_n025_coslr001_bs128_e800/certify_sigma0.25_test",
        # "amlt/smoothing/gelu_n025_coslr001_bs128_e1600/resnet152/gelu_n025_coslr001_bs128_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/eval_relu2gelu_n025_coslr01_bs128_e100/resnet152/eval_relu2gelu_n025_coslr01_bs128_e100/certify_sigma0.25_test",

        # gelu normal_resnet152, pretrain on imagenet32 with n025, train
        # "amlt/smoothing/gelu_imgn32_r152_n025_coslr1_e100/imagenet32/gelu_r152_n025_coslr1_e100/certify_sigma0.25_train",
        # "amlt/smoothing/gelu_r152_n025_coslr1_e100/imagenet32/gelu_r152_n025_coslr1_e100/certify_sigma0.25_train",
        # "amlt/smoothing/gelu_r152_n025_coslr1_e400/imagenet32/gelu_r152_n025_coslr1_e400/certify_sigma0.25_train",
        # "amlt/smoothing/gelu_r152_n025_coslr01_e200/imagenet32/gelu_r152_n025_coslr01_e200/certify_sigma0.25_train",
        # "amlt/smoothing/gelu_r152_n025_coslr01_e800/imagenet32/gelu_r152_n025_coslr01_e800/certify_sigma0.25_train",
        # "amlt/smoothing/gelu_r152_imgn32n025lr1e100_coslr001_bs128_n025_e100/finetune/gelu_r152_imgn32n025lr1e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # gelu normal_resnet152, pretrain on imagenet32 with n025, test
        # "amlt/smoothing/gelu_imgn32_r152_n025_coslr1_e100/imagenet32/gelu_r152_n025_coslr1_e100/certify_sigma0.25_test",
        # "amlt/smoothing/gelu_r152_n025_coslr1_e100/imagenet32/gelu_r152_n025_coslr1_e100/certify_sigma0.25_test",
        # "amlt/smoothing/gelu_r152_n025_coslr1_e400/imagenet32/gelu_r152_n025_coslr1_e400/certify_sigma0.25_test",
        # "amlt/smoothing/gelu_r152_n025_coslr01_e200/imagenet32/gelu_r152_n025_coslr01_e200/certify_sigma0.25_test",
        # "amlt/smoothing/gelu_r152_n025_coslr01_e800/imagenet32/gelu_r152_n025_coslr01_e800/certify_sigma0.25_test",
        # "amlt/smoothing/gelu_r152_imgn32n025lr1e100_coslr001_bs128_n025_e100/finetune/gelu_r152_imgn32n025lr1e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # nost, normal_resnet152 remove stride, keep size [32,32] untill the last global avg pooling, train from scatch on cifar10 with n025 bs64, train
        # "amlt/smoothing/nost_n025_coslr01_bs64_e100/resnet152/nost_n025_coslr01_bs64_e100/certify_sigma0.25_train",
        # "amlt/smoothing/nost_n025_coslr01_bs64_e800/resnet152/nost_n025_coslr01_bs64_e800/certify_sigma0.25_train",
        # "amlt/smoothing/nost_n025_coslr01_bs64_e1600/resnet152/nost_n025_coslr01_bs64_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/nost_n025_coslr001_bs64_e100/resnet152/nost_n025_coslr001_bs64_e100/certify_sigma0.25_train",  
        # "amlt/smoothing/nost_n025_coslr001_bs64_e800/resnet152/nost_n025_coslr001_bs64_e800/certify_sigma0.25_train",
        # "amlt/smoothing/nost_n025_coslr001_bs64_e1600/resnet152/nost_n025_coslr001_bs64_e1600/certify_sigma0.25_train",      

        # nost, normal_resnet152 remove stride, keep size [32,32] untill the last global avg pooling, train from scatch on cifar10 with n025 bs64, test
        # "amlt/smoothing/nost_n025_coslr01_bs64_e100/resnet152/nost_n025_coslr01_bs64_e100/certify_sigma0.25_test",
        # "amlt/smoothing/nost_n025_coslr01_bs64_e800/resnet152/nost_n025_coslr01_bs64_e800/certify_sigma0.25_test",
        # "amlt/smoothing/nost_n025_coslr01_bs64_e1600/resnet152/nost_n025_coslr01_bs64_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/nost_n025_coslr001_bs64_e100/resnet152/nost_n025_coslr001_bs64_e100/certify_sigma0.25_test",  
        # "amlt/smoothing/nost_n025_coslr001_bs64_e800/resnet152/nost_n025_coslr001_bs64_e800/certify_sigma0.25_test",
        # "amlt/smoothing/nost_n025_coslr001_bs64_e1600/resnet152/nost_n025_coslr001_bs64_e1600/certify_sigma0.25_test",

        # imagenet22k, pretrain n025 e100, vary lr, train
        # "amlt/smoothing/imgn22k_r152_n025_coslr1_bs1024_w4_e100/imagenet22k/imgn22k_r152_n025_coslr1_bs1024_w4_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn22k_r152_n025_coslr01_bs1024_w4_e100/imagenet22k/imgn22k_r152_n025_coslr01_bs1024_w4_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn22k_r152_n025_coslr001_bs1024_w4_e100/imagenet22k/imgn22k_r152_n025_coslr001_bs1024_w4_e100/certify_sigma0.25_train",

        # # imagenet22k, pretrain n025 e100, vary lr, test
        # "amlt/smoothing/imgn22k_r152_n025_coslr1_bs1024_w4_e100/imagenet22k/imgn22k_r152_n025_coslr1_bs1024_w4_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn22k_r152_n025_coslr01_bs1024_w4_e100/imagenet22k/imgn22k_r152_n025_coslr01_bs1024_w4_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn22k_r152_n025_coslr001_bs1024_w4_e100/imagenet22k/imgn22k_r152_n025_coslr001_bs1024_w4_e100/certify_sigma0.25_test",

        # pretrain on imagenet22k n025 lr01 e100 bs1024x8 worker4, finetune on cifar10 n025 bs128 e100, vary lr, train
        # "amlt/smoothing/r152_imgn22kn025lr01e100_coslr01_bs128_n025_e100/finetune/r152_imgn22kn025lr01e100_coslr01_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/r152_imgn22kn025lr01e100_coslr001_bs128_n025_e100/finetune/r152_imgn22kn025lr01e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # pretrain on imagenet22k n025 lr01 e100 bs1024x8 worker4, finetune on cifar10 n025 bs128 e100, vary lr, test
        # "amlt/smoothing/r152_imgn22kn025lr01e100_coslr01_bs128_n025_e100/finetune/r152_imgn22kn025lr01e100_coslr01_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/r152_imgn22kn025lr01e100_coslr001_bs128_n025_e100/finetune/r152_imgn22kn025lr01e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",

        # train from scratch on cifar10 n025 lr01 e100, certify with averaging noise, train
        # "amlt/smoothing/avgn_onlyctf_affc2/resnet152/avgn_onlyctf_affc2/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_onlyctf_affc4/resnet152/avgn_onlyctf_affc4/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_onlyctf_affc8/resnet152/avgn_onlyctf_affc8/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_onlyctf_affc16/resnet152/avgn_onlyctf_affc16/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_onlyctf_affc32/resnet152/avgn_onlyctf_affc32/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_onlyctf_befc2/resnet152/avgn_onlyctf_befc2/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_onlyctf_befc4/resnet152/avgn_onlyctf_befc4/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_onlyctf_befc8/resnet152/avgn_onlyctf_befc8/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_onlyctf_befc16/resnet152/avgn_onlyctf_befc16/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_onlyctf_befc32/resnet152/avgn_onlyctf_befc32/certify_sigma0.25_train",

        # train from scratch on cifar10 n025 lr01 e100, certify with averaging noise, test
        # "amlt/smoothing/avgn_onlyctf_affc2/resnet152/avgn_onlyctf_affc2/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_onlyctf_affc4/resnet152/avgn_onlyctf_affc4/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_onlyctf_affc8/resnet152/avgn_onlyctf_affc8/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_onlyctf_affc16/resnet152/avgn_onlyctf_affc16/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_onlyctf_affc32/resnet152/avgn_onlyctf_affc32/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_onlyctf_befc2/resnet152/avgn_onlyctf_befc2/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_onlyctf_befc4/resnet152/avgn_onlyctf_befc4/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_onlyctf_befc8/resnet152/avgn_onlyctf_befc8/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_onlyctf_befc16/resnet152/avgn_onlyctf_befc16/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_onlyctf_befc32/resnet152/avgn_onlyctf_befc32/certify_sigma0.25_test",

        # cifar10 r152, train with averaging k noise, certify without averaging noise, train
        # "amlt/smoothing/avgn_n025_lr01_bs8_affc32/resnet152/avgn_n025_lr01_bs8_affc32/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_n025_lr01_bs8_befc32/resnet152/avgn_n025_lr01_bs8_befc32/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_n025_lr01_bs16_affc16/resnet152/avgn_n025_lr01_bs16_affc16/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_n025_lr01_bs16_befc16/resnet152/avgn_n025_lr01_bs16_befc16/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_n025_lr01_bs32_affc8/resnet152/avgn_n025_lr01_bs32_affc8/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_n025_lr01_bs32_befc8/resnet152/avgn_n025_lr01_bs32_befc8/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_n025_lr01_bs64_affc4/resnet152/avgn_n025_lr01_bs64_affc4/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_n025_lr01_bs64_befc4/resnet152/avgn_n025_lr01_bs64_befc4/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_n025_lr01_bs128_affc2/resnet152/avgn_n025_lr01_bs128_affc2/certify_sigma0.25_train",
        # "amlt/smoothing/avgn_n025_lr01_bs128_befc2/resnet152/avgn_n025_lr01_bs128_befc2/certify_sigma0.25_train",

        #  cifar10 r152, train with averaging k noise, certify without averaging noise, test
        # "amlt/smoothing/avgn_n025_lr01_bs128_affc2/resnet152/avgn_n025_lr01_bs128_affc2/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_n025_lr01_bs128_befc2/resnet152/avgn_n025_lr01_bs128_befc2/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_n025_lr01_bs64_affc4/resnet152/avgn_n025_lr01_bs64_affc4/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_n025_lr01_bs64_befc4/resnet152/avgn_n025_lr01_bs64_befc4/certify_sigma0.25_test",     
        # "amlt/smoothing/avgn_n025_lr01_bs32_affc8/resnet152/avgn_n025_lr01_bs32_affc8/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_n025_lr01_bs32_befc8/resnet152/avgn_n025_lr01_bs32_befc8/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_n025_lr01_bs16_affc16/resnet152/avgn_n025_lr01_bs16_affc16/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_n025_lr01_bs16_befc16/resnet152/avgn_n025_lr01_bs16_befc16/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_n025_lr01_bs8_affc32/resnet152/avgn_n025_lr01_bs8_affc32/certify_sigma0.25_test",
        # "amlt/smoothing/avgn_n025_lr01_bs8_befc32/resnet152/avgn_n025_lr01_bs8_befc32/certify_sigma0.25_test",

        #  cifar10 r152, only certify, average input noise, train
        # "amlt/smoothing/avgin2_onlyctf/resnet152/avgin2_onlyctf/certify_sigma0.25_train",
        # "amlt/smoothing/avgin4_onlyctf/resnet152/avgin4_onlyctf/certify_sigma0.25_train",
        # "amlt/smoothing/avgin8_onlyctf/resnet152/avgin8_onlyctf/certify_sigma0.25_train",
        # "amlt/smoothing/avgin16_onlyctf/resnet152/avgin16_onlyctf/certify_sigma0.25_train",
        # "amlt/smoothing/avgin32_onlyctf/resnet152/avgin32_onlyctf/certify_sigma0.25_train",

        #  cifar10 r152, only certify, average input noise, test
        # "amlt/smoothing/avgin2_onlyctf/resnet152/avgin2_onlyctf/certify_sigma0.25_test",
        # "amlt/smoothing/avgin4_onlyctf/resnet152/avgin4_onlyctf/certify_sigma0.25_test",
        # "amlt/smoothing/avgin8_onlyctf/resnet152/avgin8_onlyctf/certify_sigma0.25_test",
        # "amlt/smoothing/avgin16_onlyctf/resnet152/avgin16_onlyctf/certify_sigma0.25_test",
        # "amlt/smoothing/avgin32_onlyctf/resnet152/avgin32_onlyctf/certify_sigma0.25_test",

        # cifar10 r152, train with average input noise, normal certify, train
        # "amlt/smoothing/avgin2_n025_lr01_e100/resnet152/avgin2_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/avgin4_n025_lr01_e100/resnet152/avgin4_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/avgin8_n025_lr01_e100/resnet152/avgin8_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/avgin16_n025_lr01_e100/resnet152/avgin16_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/avgin32_n025_lr01_e100/resnet152/avgin32_n025_lr01_e100/certify_sigma0.25_train",

        # cifar10 r152, train with average input noise, normal certify, test
        # "amlt/smoothing/avgin2_n025_lr01_e100/resnet152/avgin2_n025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/avgin4_n025_lr01_e100/resnet152/avgin4_n025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/avgin8_n025_lr01_e100/resnet152/avgin8_n025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/avgin16_n025_lr01_e100/resnet152/avgin16_n025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/avgin32_n025_lr01_e100/resnet152/avgin32_n025_lr01_e100/certify_sigma0.25_test",

        # cifar10 r152, train with average input noise, certify with average input noise, train
        # "amlt/smoothing/avgin2ctf_n025_lr01_e100/resnet152/avgin2ctf_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/avgin4ctf_n025_lr01_e100/resnet152/avgin4ctf_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/avgin8ctf_n025_lr01_e100/resnet152/avgin8ctf_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/avgin16ctf_n025_lr01_e100/resnet152/avgin16ctf_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/avgin32ctf_n025_lr01_e100/resnet152/avgin32ctf_n025_lr01_e100/certify_sigma0.25_train",

        # cifar10 r152, add extra noise for f, train
        # "amlt/smoothing/fn1a2_gn025_lr01_bs128/resnet152/fn1a2_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn05a2_gn025_lr01_bs128/resnet152/fn05a2_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn01a2_gn025_lr01_bs128/resnet152/fn01a2_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a2_gn025_lr01_bs128/resnet152/fn005a2_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a2_gn025_lr01_bs128/resnet152/fn001a2_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn0005a2_gn025_lr01_bs128/resnet152/fn0005a2_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn0001a2_gn025_lr01_bs128/resnet152/fn0001a2_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn00001a2_gn025_lr01_bs128/resnet152/fn00001a2_gn025_lr01_bs128/certify_sigma0.25_train",

        # "amlt/smoothing/fn1a4_gn025_lr01_bs128/resnet152/fn1a4_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn05a4_gn025_lr01_bs128/resnet152/fn05a4_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn01a4_gn025_lr01_bs128/resnet152/fn01a4_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a4_gn025_lr01_bs128/resnet152/fn005a4_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a4_gn025_lr01_bs128/resnet152/fn001a4_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn0005a4_gn025_lr01_bs128/resnet152/fn0005a4_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn0001a4_gn025_lr01_bs128/resnet152/fn0001a4_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn00001a4_gn025_lr01_bs128/resnet152/fn00001a4_gn025_lr01_bs128/certify_sigma0.25_train",

        # "amlt/smoothing/fn1a8_gn025_lr01_bs128/resnet152/fn1a8_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn05a8_gn025_lr01_bs128/resnet152/fn05a8_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn01a8_gn025_lr01_bs128/resnet152/fn01a8_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a8_gn025_lr01_bs128/resnet152/fn005a8_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a8_gn025_lr01_bs128/resnet152/fn001a8_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn0005a8_gn025_lr01_bs128/resnet152/fn0005a8_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn0001a8_gn025_lr01_bs128/resnet152/fn0001a8_gn025_lr01_bs128/certify_sigma0.25_train",
        # "amlt/smoothing/fn00001a8_gn025_lr01_bs128/resnet152/fn00001a8_gn025_lr01_bs128/certify_sigma0.25_train",

        # "amlt/smoothing/fn1a16_gn025_lr01_bs64/resnet152/fn1a16_gn025_lr01_bs64/certify_sigma0.25_train",
        # "amlt/smoothing/fn05a16_gn025_lr01_bs64/resnet152/fn05a16_gn025_lr01_bs64/certify_sigma0.25_train",
        # "amlt/smoothing/fn01a16_gn025_lr01_bs64/resnet152/fn01a16_gn025_lr01_bs64/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a16_gn025_lr01_bs64/resnet152/fn005a16_gn025_lr01_bs64/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a16_gn025_lr01_bs64/resnet152/fn001a16_gn025_lr01_bs64/certify_sigma0.25_train",
        # "amlt/smoothing/fn0005a16_gn025_lr01_bs64/resnet152/fn0005a16_gn025_lr01_bs64/certify_sigma0.25_train",
        # "amlt/smoothing/fn0001a16_gn025_lr01_bs64/resnet152/fn0001a16_gn025_lr01_bs64/certify_sigma0.25_train",
        # "amlt/smoothing/fn00001a16_gn025_lr01_bs64/resnet152/fn00001a16_gn025_lr01_bs64/certify_sigma0.25_train",

        # "amlt/smoothing/fn1a32_gn025_lr01_bs32/resnet152/fn1a32_gn025_lr01_bs32/certify_sigma0.25_train",
        # "amlt/smoothing/fn05a32_gn025_lr01_bs32/resnet152/fn05a32_gn025_lr01_bs32/certify_sigma0.25_train",
        # "amlt/smoothing/fn01a32_gn025_lr01_bs32/resnet152/fn01a32_gn025_lr01_bs32/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a32_gn025_lr01_bs32/resnet152/fn005a32_gn025_lr01_bs32/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a32_gn025_lr01_bs32/resnet152/fn001a32_gn025_lr01_bs32/certify_sigma0.25_train",
        # "amlt/smoothing/fn0005a32_gn025_lr01_bs32/resnet152/fn0005a32_gn025_lr01_bs32/certify_sigma0.25_train",
        # "amlt/smoothing/fn0001a32_gn025_lr01_bs32/resnet152/fn0001a32_gn025_lr01_bs32/certify_sigma0.25_train",
        # "amlt/smoothing/fn00001a32_gn025_lr01_bs32/resnet152/fn00001a32_gn025_lr01_bs32/certify_sigma0.25_train",

        # "amlt/smoothing/fn005a16_gn025_lr001_e100/resnet152/fn005a16_gn025_lr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a16_gn025_lr001_e800/resnet152/fn005a16_gn025_lr001_e800/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a16_gn025_lr001_e1600/resnet152/fn005a16_gn025_lr001_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a16_gn025_lr01_bs64/resnet152/fn005a16_gn025_lr01_bs64/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a16_gn025_lr01_e800/resnet152/fn005a16_gn025_lr01_e800/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a16_gn025_lr01_e1600/resnet152/fn005a16_gn025_lr01_e1600/certify_sigma0.25_train",

        # "amlt/smoothing/fn005a32_gn025_lr001_e100/resnet152/fn005a32_gn025_lr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a32_gn025_lr001_e800/resnet152/fn005a32_gn025_lr001_e800/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a32_gn025_lr001_e1600/resnet152/fn005a32_gn025_lr001_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a32_gn025_lr01_bs32/resnet152/fn005a32_gn025_lr01_bs32/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a32_gn025_lr01_e800/resnet152/fn005a32_gn025_lr01_e800/certify_sigma0.25_train",
        # "amlt/smoothing/fn005a32_gn025_lr01_e1600/resnet152/fn005a32_gn025_lr01_e1600/certify_sigma0.25_train",

        # "amlt/smoothing/fn001a16_gn025_lr001_e100/resnet152/fn001a16_gn025_lr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a16_gn025_lr001_e800/resnet152/fn001a16_gn025_lr001_e800/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a16_gn025_lr001_e1600/resnet152/fn001a16_gn025_lr001_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a16_gn025_lr01_bs64/resnet152/fn001a16_gn025_lr01_bs64/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a16_gn025_lr01_e800/resnet152/fn001a16_gn025_lr01_e800/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a16_gn025_lr01_e1600/resnet152/fn001a16_gn025_lr01_e1600/certify_sigma0.25_train",

        # "amlt/smoothing/fn001a32_gn025_lr001_e100/resnet152/fn001a32_gn025_lr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a32_gn025_lr001_e800/resnet152/fn001a32_gn025_lr001_e800/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a32_gn025_lr001_e1600/resnet152/fn001a32_gn025_lr001_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a32_gn025_lr01_bs32/resnet152/fn001a32_gn025_lr01_bs32/certify_sigma0.25_train",
        # "amlt/smoothing/fn001a32_gn025_lr01_e800/resnet152/fn001a32_gn025_lr01_e800/certify_sigma0.25_train",
        # # "amlt/smoothing/fn001a32_gn025_lr01_e1600/resnet152/fn001a32_gn025_lr01_e1600/certify_sigma0.25_train",


        # cifar10 r152, add extra noise for f, test
        # "amlt/smoothing/fn1a2_gn025_lr01_bs128/resnet152/fn1a2_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn05a2_gn025_lr01_bs128/resnet152/fn05a2_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn01a2_gn025_lr01_bs128/resnet152/fn01a2_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a2_gn025_lr01_bs128/resnet152/fn005a2_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a2_gn025_lr01_bs128/resnet152/fn001a2_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn0005a2_gn025_lr01_bs128/resnet152/fn0005a2_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn0001a2_gn025_lr01_bs128/resnet152/fn0001a2_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn00001a2_gn025_lr01_bs128/resnet152/fn00001a2_gn025_lr01_bs128/certify_sigma0.25_test",

        # "amlt/smoothing/fn1a4_gn025_lr01_bs128/resnet152/fn1a4_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn05a4_gn025_lr01_bs128/resnet152/fn05a4_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn01a4_gn025_lr01_bs128/resnet152/fn01a4_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a4_gn025_lr01_bs128/resnet152/fn005a4_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a4_gn025_lr01_bs128/resnet152/fn001a4_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn0005a4_gn025_lr01_bs128/resnet152/fn0005a4_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn0001a4_gn025_lr01_bs128/resnet152/fn0001a4_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn00001a4_gn025_lr01_bs128/resnet152/fn00001a4_gn025_lr01_bs128/certify_sigma0.25_test",

        # "amlt/smoothing/fn1a8_gn025_lr01_bs128/resnet152/fn1a8_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn05a8_gn025_lr01_bs128/resnet152/fn05a8_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn01a8_gn025_lr01_bs128/resnet152/fn01a8_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a8_gn025_lr01_bs128/resnet152/fn005a8_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a8_gn025_lr01_bs128/resnet152/fn001a8_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn0005a8_gn025_lr01_bs128/resnet152/fn0005a8_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn0001a8_gn025_lr01_bs128/resnet152/fn0001a8_gn025_lr01_bs128/certify_sigma0.25_test",
        # "amlt/smoothing/fn00001a8_gn025_lr01_bs128/resnet152/fn00001a8_gn025_lr01_bs128/certify_sigma0.25_test",

        # "amlt/smoothing/fn1a16_gn025_lr01_bs64/resnet152/fn1a16_gn025_lr01_bs64/certify_sigma0.25_test",
        # "amlt/smoothing/fn05a16_gn025_lr01_bs64/resnet152/fn05a16_gn025_lr01_bs64/certify_sigma0.25_test",
        # "amlt/smoothing/fn01a16_gn025_lr01_bs64/resnet152/fn01a16_gn025_lr01_bs64/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a16_gn025_lr01_bs64/resnet152/fn005a16_gn025_lr01_bs64/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a16_gn025_lr01_bs64/resnet152/fn001a16_gn025_lr01_bs64/certify_sigma0.25_test",
        # "amlt/smoothing/fn0005a16_gn025_lr01_bs64/resnet152/fn0005a16_gn025_lr01_bs64/certify_sigma0.25_test",
        # "amlt/smoothing/fn0001a16_gn025_lr01_bs64/resnet152/fn0001a16_gn025_lr01_bs64/certify_sigma0.25_test",
        # "amlt/smoothing/fn00001a16_gn025_lr01_bs64/resnet152/fn00001a16_gn025_lr01_bs64/certify_sigma0.25_test",

        # "amlt/smoothing/fn1a32_gn025_lr01_bs32/resnet152/fn1a32_gn025_lr01_bs32/certify_sigma0.25_test",
        # # "amlt/smoothing/fn05a32_gn025_lr01_bs32/resnet152/fn05a32_gn025_lr01_bs32/certify_sigma0.25_test",
        # "amlt/smoothing/fn01a32_gn025_lr01_bs32/resnet152/fn01a32_gn025_lr01_bs32/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a32_gn025_lr01_bs32/resnet152/fn005a32_gn025_lr01_bs32/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a32_gn025_lr01_bs32/resnet152/fn001a32_gn025_lr01_bs32/certify_sigma0.25_test",
        # "amlt/smoothing/fn0005a32_gn025_lr01_bs32/resnet152/fn0005a32_gn025_lr01_bs32/certify_sigma0.25_test",
        # "amlt/smoothing/fn0001a32_gn025_lr01_bs32/resnet152/fn0001a32_gn025_lr01_bs32/certify_sigma0.25_test",
        # "amlt/smoothing/fn00001a32_gn025_lr01_bs32/resnet152/fn00001a32_gn025_lr01_bs32/certify_sigma0.25_test",

        # "amlt/smoothing/fn005a16_gn025_lr001_e100/resnet152/fn005a16_gn025_lr001_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a16_gn025_lr001_e800/resnet152/fn005a16_gn025_lr001_e800/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a16_gn025_lr001_e1600/resnet152/fn005a16_gn025_lr001_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a16_gn025_lr01_bs64/resnet152/fn005a16_gn025_lr01_bs64/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a16_gn025_lr01_e800/resnet152/fn005a16_gn025_lr01_e800/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a16_gn025_lr01_e1600/resnet152/fn005a16_gn025_lr01_e1600/certify_sigma0.25_test",

        # "amlt/smoothing/fn005a32_gn025_lr001_e100/resnet152/fn005a32_gn025_lr001_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a32_gn025_lr001_e800/resnet152/fn005a32_gn025_lr001_e800/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a32_gn025_lr001_e1600/resnet152/fn005a32_gn025_lr001_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a32_gn025_lr01_bs32/resnet152/fn005a32_gn025_lr01_bs32/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a32_gn025_lr01_e800/resnet152/fn005a32_gn025_lr01_e800/certify_sigma0.25_test",
        # "amlt/smoothing/fn005a32_gn025_lr01_e1600/resnet152/fn005a32_gn025_lr01_e1600/certify_sigma0.25_test",

        # "amlt/smoothing/fn001a16_gn025_lr001_e100/resnet152/fn001a16_gn025_lr001_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a16_gn025_lr001_e800/resnet152/fn001a16_gn025_lr001_e800/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a16_gn025_lr001_e1600/resnet152/fn001a16_gn025_lr001_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a16_gn025_lr01_bs64/resnet152/fn001a16_gn025_lr01_bs64/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a16_gn025_lr01_e800/resnet152/fn001a16_gn025_lr01_e800/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a16_gn025_lr01_e1600/resnet152/fn001a16_gn025_lr01_e1600/certify_sigma0.25_test",

        # "amlt/smoothing/fn001a32_gn025_lr001_e100/resnet152/fn001a32_gn025_lr001_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a32_gn025_lr001_e800/resnet152/fn001a32_gn025_lr001_e800/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a32_gn025_lr001_e1600/resnet152/fn001a32_gn025_lr001_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a32_gn025_lr01_bs32/resnet152/fn001a32_gn025_lr01_bs32/certify_sigma0.25_test",
        # "amlt/smoothing/fn001a32_gn025_lr01_e800/resnet152/fn001a32_gn025_lr01_e800/certify_sigma0.25_test",
        # # "amlt/smoothing/fn001a32_gn025_lr01_e1600/resnet152/fn001a32_gn025_lr01_e1600/certify_sigma0.25_test",


        # attention pool fnoise, vary fnoise, vary noise num, train from scratch on cifar10
        # "amlt/smoothing/atph16fc_fn01a2_gn025_lr01_e100/resnet152/atph16fc_fn01a2_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a4_gn025_lr01_e100/resnet152/atph16fc_fn01a4_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a8_gn025_lr01_e100/resnet152/atph16fc_fn01a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr01_e100/resnet152/atph16fc_fn01a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr01_e100/resnet152/atph16fc_fn01a32_gn025_lr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/atph16fc_fn005a2_gn025_lr01_e100/resnet152/atph16fc_fn005a2_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a4_gn025_lr01_e100/resnet152/atph16fc_fn005a4_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a8_gn025_lr01_e100/resnet152/atph16fc_fn005a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr01_e100/resnet152/atph16fc_fn005a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr01_e100/resnet152/atph16fc_fn005a32_gn025_lr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/atph16fc_fn001a2_gn025_lr01_e100/resnet152/atph16fc_fn001a2_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn001a4_gn025_lr01_e100/resnet152/atph16fc_fn001a4_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn001a8_gn025_lr01_e100/resnet152/atph16fc_fn001a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn001a16_gn025_lr01_e100/resnet152/atph16fc_fn001a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn001a32_gn025_lr01_e100/resnet152/atph16fc_fn001a32_gn025_lr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/atph16fc_fn0005a2_gn025_lr01_e100/resnet152/atph16fc_fn0005a2_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn0005a4_gn025_lr01_e100/resnet152/atph16fc_fn0005a4_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn0005a8_gn025_lr01_e100/resnet152/atph16fc_fn0005a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn0005a16_gn025_lr01_e100/resnet152/atph16fc_fn0005a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn0005a32_gn025_lr01_e100/resnet152/atph16fc_fn0005a32_gn025_lr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/atph16fc_fn0001a2_gn025_lr01_e100/resnet152/atph16fc_fn0001a2_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn0001a4_gn025_lr01_e100/resnet152/atph16fc_fn0001a4_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn0001a8_gn025_lr01_e100/resnet152/atph16fc_fn0001a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn0001a16_gn025_lr01_e100/resnet152/atph16fc_fn0001a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn0001a32_gn025_lr01_e100/resnet152/atph16fc_fn0001a32_gn025_lr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/atph16fc_fn01a2_gn025_lr01_e100/resnet152/atph16fc_fn01a2_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a4_gn025_lr01_e100/resnet152/atph16fc_fn01a4_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a8_gn025_lr01_e100/resnet152/atph16fc_fn01a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr01_e100/resnet152/atph16fc_fn01a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr01_e100/resnet152/atph16fc_fn01a32_gn025_lr01_e100/certify_sigma0.25_test",

        # "amlt/smoothing/atph16fc_fn005a2_gn025_lr01_e100/resnet152/atph16fc_fn005a2_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a4_gn025_lr01_e100/resnet152/atph16fc_fn005a4_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a8_gn025_lr01_e100/resnet152/atph16fc_fn005a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr01_e100/resnet152/atph16fc_fn005a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr01_e100/resnet152/atph16fc_fn005a32_gn025_lr01_e100/certify_sigma0.25_test",

        # "amlt/smoothing/atph16fc_fn001a2_gn025_lr01_e100/resnet152/atph16fc_fn001a2_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn001a4_gn025_lr01_e100/resnet152/atph16fc_fn001a4_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn001a8_gn025_lr01_e100/resnet152/atph16fc_fn001a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn001a16_gn025_lr01_e100/resnet152/atph16fc_fn001a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn001a32_gn025_lr01_e100/resnet152/atph16fc_fn001a32_gn025_lr01_e100/certify_sigma0.25_test",

        # "amlt/smoothing/atph16fc_fn0005a2_gn025_lr01_e100/resnet152/atph16fc_fn0005a2_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn0005a4_gn025_lr01_e100/resnet152/atph16fc_fn0005a4_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn0005a8_gn025_lr01_e100/resnet152/atph16fc_fn0005a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn0005a16_gn025_lr01_e100/resnet152/atph16fc_fn0005a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn0005a32_gn025_lr01_e100/resnet152/atph16fc_fn0005a32_gn025_lr01_e100/certify_sigma0.25_test",

        # "amlt/smoothing/atph16fc_fn0001a2_gn025_lr01_e100/resnet152/atph16fc_fn0001a2_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn0001a4_gn025_lr01_e100/resnet152/atph16fc_fn0001a4_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn0001a8_gn025_lr01_e100/resnet152/atph16fc_fn0001a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn0001a16_gn025_lr01_e100/resnet152/atph16fc_fn0001a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn0001a32_gn025_lr01_e100/resnet152/atph16fc_fn0001a32_gn025_lr01_e100/certify_sigma0.25_test",


        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr01_e100/resnet152/atph16fc_fn01a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr01_e800/resnet152/atph16fc_fn01a16_gn025_lr01_e800/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr01_e1600/resnet152/atph16fc_fn01a16_gn025_lr01_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr001_e100/resnet152/atph16fc_fn01a16_gn025_lr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr001_e800/resnet152/atph16fc_fn01a16_gn025_lr001_e800/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr001_e1600/resnet152/atph16fc_fn01a16_gn025_lr001_e1600/certify_sigma0.25_train",

        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr01_e100/resnet152/atph16fc_fn01a32_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr01_e800/resnet152/atph16fc_fn01a32_gn025_lr01_e800/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr01_e1600/resnet152/atph16fc_fn01a32_gn025_lr01_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr001_e100/resnet152/atph16fc_fn01a32_gn025_lr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr001_e800/resnet152/atph16fc_fn01a32_gn025_lr001_e800/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr001_e1600/resnet152/atph16fc_fn01a32_gn025_lr001_e1600/certify_sigma0.25_train",

        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr01_e100/resnet152/atph16fc_fn005a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr01_e800/resnet152/atph16fc_fn005a16_gn025_lr01_e800/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr01_e1600/resnet152/atph16fc_fn005a16_gn025_lr01_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr001_e100/resnet152/atph16fc_fn005a16_gn025_lr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr001_e800/resnet152/atph16fc_fn005a16_gn025_lr001_e800/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr001_e1600/resnet152/atph16fc_fn005a16_gn025_lr001_e1600/certify_sigma0.25_train",

        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr01_e100/resnet152/atph16fc_fn005a32_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr01_e800/resnet152/atph16fc_fn005a32_gn025_lr01_e800/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr01_e1600/resnet152/atph16fc_fn005a32_gn025_lr01_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr001_e100/resnet152/atph16fc_fn005a32_gn025_lr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr001_e800/resnet152/atph16fc_fn005a32_gn025_lr001_e800/certify_sigma0.25_train",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr001_e1600/resnet152/atph16fc_fn005a32_gn025_lr001_e1600/certify_sigma0.25_train",

        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr01_e100/resnet152/atph16fc_fn01a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr01_e800/resnet152/atph16fc_fn01a16_gn025_lr01_e800/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr01_e1600/resnet152/atph16fc_fn01a16_gn025_lr01_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr001_e100/resnet152/atph16fc_fn01a16_gn025_lr001_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr001_e800/resnet152/atph16fc_fn01a16_gn025_lr001_e800/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a16_gn025_lr001_e1600/resnet152/atph16fc_fn01a16_gn025_lr001_e1600/certify_sigma0.25_test",

        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr01_e100/resnet152/atph16fc_fn01a32_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr01_e800/resnet152/atph16fc_fn01a32_gn025_lr01_e800/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr01_e1600/resnet152/atph16fc_fn01a32_gn025_lr01_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr001_e100/resnet152/atph16fc_fn01a32_gn025_lr001_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr001_e800/resnet152/atph16fc_fn01a32_gn025_lr001_e800/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn01a32_gn025_lr001_e1600/resnet152/atph16fc_fn01a32_gn025_lr001_e1600/certify_sigma0.25_test",

        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr01_e100/resnet152/atph16fc_fn005a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr01_e800/resnet152/atph16fc_fn005a16_gn025_lr01_e800/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr01_e1600/resnet152/atph16fc_fn005a16_gn025_lr01_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr001_e100/resnet152/atph16fc_fn005a16_gn025_lr001_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr001_e800/resnet152/atph16fc_fn005a16_gn025_lr001_e800/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a16_gn025_lr001_e1600/resnet152/atph16fc_fn005a16_gn025_lr001_e1600/certify_sigma0.25_test",

        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr01_e100/resnet152/atph16fc_fn005a32_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr01_e800/resnet152/atph16fc_fn005a32_gn025_lr01_e800/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr01_e1600/resnet152/atph16fc_fn005a32_gn025_lr01_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr001_e100/resnet152/atph16fc_fn005a32_gn025_lr001_e100/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr001_e800/resnet152/atph16fc_fn005a32_gn025_lr001_e800/certify_sigma0.25_test",
        # "amlt/smoothing/atph16fc_fn005a32_gn025_lr001_e1600/resnet152/atph16fc_fn005a32_gn025_lr001_e1600/certify_sigma0.25_test",


        # vit, cifar10 n025, train from scratch, train
        # "amlt/smoothing/vit_cf10_n025_coslr003_e100/vit/vit_cf10_n025_coslr003_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_cf10_n025_coslr0003_e100/vit/vit_cf10_n025_coslr0003_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_cf10_n025_coslr00003_e100/vit/vit_cf10_n025_coslr00003_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_cf10_n025_coslr000003_e100/vit/vit_cf10_n025_coslr000003_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_cf10_n025_coslr0000003_e100/vit/vit_cf10_n025_coslr0000003_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_cf10_n025_coslr00003_e200/vit/vit_cf10_n025_coslr00003_e200/certify_sigma0.25_train",
        # "amlt/smoothing/vit_cf10_n025_coslr0003_e300/vit/vit_cf10_n025_coslr0003_e300/certify_sigma0.25_train",
        # "amlt/smoothing/vit_cf10_n025_coslr00003_e400/vit/vit_cf10_n025_coslr00003_e400/certify_sigma0.25_train",
        # "amlt/smoothing/vit_cf10_n025_coslr00003_e800/vit/vit_cf10_n025_coslr00003_e800/certify_sigma0.25_train",

        # vit, cifar10 n025, train from scratch, test
        # "amlt/smoothing/vit_cf10_n025_coslr003_e100/vit/vit_cf10_n025_coslr003_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_cf10_n025_coslr0003_e100/vit/vit_cf10_n025_coslr0003_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_cf10_n025_coslr00003_e100/vit/vit_cf10_n025_coslr00003_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_cf10_n025_coslr000003_e100/vit/vit_cf10_n025_coslr000003_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_cf10_n025_coslr0000003_e100/vit/vit_cf10_n025_coslr0000003_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_cf10_n025_coslr00003_e200/vit/vit_cf10_n025_coslr00003_e200/certify_sigma0.25_test",
        # "amlt/smoothing/vit_cf10_n025_coslr0003_e300/vit/vit_cf10_n025_coslr0003_e300/certify_sigma0.25_test",
        # "amlt/smoothing/vit_cf10_n025_coslr00003_e400/vit/vit_cf10_n025_coslr00003_e400/certify_sigma0.25_test",
        # "amlt/smoothing/vit_cf10_n025_coslr00003_e800/vit/vit_cf10_n025_coslr00003_e800/certify_sigma0.25_test",

        # vit, imagenet32 n025, train
        # "amlt/smoothing/vit_imgn32_n025_coslr003_e100/vit/vit_imgn32_n025_coslr003_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_imgn32_n025_coslr000003_e100/vit/vit_imgn32_n025_coslr000003_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_imgn32_n025_coslr0000003_e100/vit/vit_imgn32_n025_coslr0000003_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_imgn32_n025_coslr0003_e100/vit/vit_imgn32_n025_coslr0003_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_imgn32_n025_coslr0003_e200/vit/vit_imgn32_n025_coslr0003_e200/certify_sigma0.25_train",
        # "amlt/smoothing/vit_imgn32_n025_coslr0003_e300/vit/vit_imgn32_n025_coslr0003_e300/certify_sigma0.25_train",
        # "amlt/smoothing/vit_imgn32_n025_coslr0003_e400/vit/vit_imgn32_n025_coslr0003_e400/certify_sigma0.25_train",
        # "amlt/smoothing/vit_imgn32_n025_coslr0003_e800/vit/vit_imgn32_n025_coslr0003_e800/certify_sigma0.25_train",
        # "amlt/smoothing/vit_imgn32_n025_coslr00003_e100/vit/vit_imgn32_n025_coslr00003_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_imgn32_n025_coslr00003_e200/vit/vit_imgn32_n025_coslr00003_e200/certify_sigma0.25_train",
        # "amlt/smoothing/vit_imgn32_n025_coslr00003_e400/vit/vit_imgn32_n025_coslr00003_e400/certify_sigma0.25_train",
        # "amlt/smoothing/vit_imgn32_n025_coslr00003_e800/vit/vit_imgn32_n025_coslr00003_e800/certify_sigma0.25_train",


        # vit, imagenet32 n025, test
        # # "amlt/smoothing/vit_imgn32_n025_coslr003_e100/vit/vit_imgn32_n025_coslr003_e100/certify_sigma0.25_test",
        # # "amlt/smoothing/vit_imgn32_n025_coslr000003_e100/vit/vit_imgn32_n025_coslr000003_e100/certify_sigma0.25_test",
        # # "amlt/smoothing/vit_imgn32_n025_coslr0000003_e100/vit/vit_imgn32_n025_coslr0000003_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_imgn32_n025_coslr0003_e100/vit/vit_imgn32_n025_coslr0003_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_imgn32_n025_coslr0003_e200/vit/vit_imgn32_n025_coslr0003_e200/certify_sigma0.25_test",
        # # "amlt/smoothing/vit_imgn32_n025_coslr0003_e300/vit/vit_imgn32_n025_coslr0003_e300/certify_sigma0.25_test",
        # "amlt/smoothing/vit_imgn32_n025_coslr0003_e400/vit/vit_imgn32_n025_coslr0003_e400/certify_sigma0.25_test",
        # "amlt/smoothing/vit_imgn32_n025_coslr0003_e800/vit/vit_imgn32_n025_coslr0003_e800/certify_sigma0.25_test",
        # "amlt/smoothing/vit_imgn32_n025_coslr00003_e100/vit/vit_imgn32_n025_coslr00003_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_imgn32_n025_coslr00003_e200/vit/vit_imgn32_n025_coslr00003_e200/certify_sigma0.25_test",
        # "amlt/smoothing/vit_imgn32_n025_coslr00003_e400/vit/vit_imgn32_n025_coslr00003_e400/certify_sigma0.25_test",
        # "amlt/smoothing/vit_imgn32_n025_coslr00003_e800/vit/vit_imgn32_n025_coslr00003_e800/certify_sigma0.25_test",

        # baseline
        # "amlt/smoothing/baseline_r152_n025_lr01_e100/resnet152/baseline_r152_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/baseline_cp2_r152_n025_lr01_e100/resnet152/baseline_cp2_r152_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/baseline_cp3_r152_n025_lr01_e100/resnet152/baseline_cp3_r152_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/baseline_cp4_r152_n025_lr01_e100/resnet152/baseline_cp4_r152_n025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/baseline_cp5_r152_n025_lr01_e100/resnet152/baseline_cp5_r152_n025_lr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/baseline_r152_n025_lr01_e100/resnet152/baseline_r152_n025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/baseline_cp2_r152_n025_lr01_e100/resnet152/baseline_cp2_r152_n025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/baseline_cp3_r152_n025_lr01_e100/resnet152/baseline_cp3_r152_n025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/baseline_cp4_r152_n025_lr01_e100/resnet152/baseline_cp4_r152_n025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/baseline_cp5_r152_n025_lr01_e100/resnet152/baseline_cp5_r152_n025_lr01_e100/certify_sigma0.25_test",

        # train and certify with different noise
        # "amlt/smoothing/tn025_cn025_lr01_bs128_e100/resnet152/tn025_cn025_lr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/tn025_cn025_lr01_bs128_e800/resnet152/tn025_cn025_lr01_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/tn025_cn025_lr01_bs128_e1600/resnet152/tn025_cn025_lr01_bs128_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/tn02_cn025_lr01_bs128_e100/resnet152/tn02_cn025_lr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/tn02_cn025_lr01_bs128_e800/resnet152/tn02_cn025_lr01_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/tn02_cn025_lr01_bs128_e1600/resnet152/tn02_cn025_lr01_bs128_e1600/certify_sigma0.25_train",        
        # "amlt/smoothing/tn03_cn025_lr01_bs128_e100/resnet152/tn03_cn025_lr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/tn03_cn025_lr01_bs128_e800/resnet152/tn03_cn025_lr01_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/tn03_cn025_lr01_bs128_e1600/resnet152/tn03_cn025_lr01_bs128_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/tn04_cn025_lr01_bs128_e100/resnet152/tn04_cn025_lr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/tn04_cn025_lr01_bs128_e800/resnet152/tn04_cn025_lr01_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/tn04_cn025_lr01_bs128_e1600/resnet152/tn04_cn025_lr01_bs128_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/tn05_cn025_lr01_bs128_e100/resnet152/tn05_cn025_lr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/tn05_cn025_lr01_bs128_e800/resnet152/tn05_cn025_lr01_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/tn05_cn025_lr01_bs128_e1600/resnet152/tn05_cn025_lr01_bs128_e1600/certify_sigma0.25_train",

        # "amlt/smoothing/tn02_cn025_lr001_bs128_e100/resnet152/tn02_cn025_lr001_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/tn02_cn025_lr001_bs128_e800/resnet152/tn02_cn025_lr001_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/tn02_cn025_lr001_bs128_e1600/resnet152/tn02_cn025_lr001_bs128_e1600/certify_sigma0.25_train",

        # "amlt/smoothing/tn03_cn025_lr001_bs128_e100/resnet152/tn03_cn025_lr001_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/tn03_cn025_lr001_bs128_e800/resnet152/tn03_cn025_lr001_bs128_e800/certify_sigma0.25_train",
        # "amlt/smoothing/tn03_cn025_lr001_bs128_e1600/resnet152/tn03_cn025_lr001_bs128_e1600/certify_sigma0.25_train",


        # "amlt/smoothing/tn025_cn025_lr01_bs128_e100/resnet152/tn025_cn025_lr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tn025_cn025_lr01_bs128_e800/resnet152/tn025_cn025_lr01_bs128_e800/certify_sigma0.25_test",
        # "amlt/smoothing/tn025_cn025_lr01_bs128_e1600/resnet152/tn025_cn025_lr01_bs128_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/tn02_cn025_lr01_bs128_e100/resnet152/tn02_cn025_lr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tn02_cn025_lr01_bs128_e800/resnet152/tn02_cn025_lr01_bs128_e800/certify_sigma0.25_test",
        # "amlt/smoothing/tn02_cn025_lr01_bs128_e1600/resnet152/tn02_cn025_lr01_bs128_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/tn03_cn025_lr01_bs128_e100/resnet152/tn03_cn025_lr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tn03_cn025_lr01_bs128_e800/resnet152/tn03_cn025_lr01_bs128_e800/certify_sigma0.25_test",
        # "amlt/smoothing/tn03_cn025_lr01_bs128_e1600/resnet152/tn03_cn025_lr01_bs128_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/tn04_cn025_lr01_bs128_e100/resnet152/tn04_cn025_lr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tn04_cn025_lr01_bs128_e800/resnet152/tn04_cn025_lr01_bs128_e800/certify_sigma0.25_test",
        # "amlt/smoothing/tn04_cn025_lr01_bs128_e1600/resnet152/tn04_cn025_lr01_bs128_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/tn05_cn025_lr01_bs128_e100/resnet152/tn05_cn025_lr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tn05_cn025_lr01_bs128_e800/resnet152/tn05_cn025_lr01_bs128_e800/certify_sigma0.25_test",
        # "amlt/smoothing/tn05_cn025_lr01_bs128_e1600/resnet152/tn05_cn025_lr01_bs128_e1600/certify_sigma0.25_test",

        # "amlt/smoothing/tn02_cn025_lr001_bs128_e100/resnet152/tn02_cn025_lr001_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tn02_cn025_lr001_bs128_e800/resnet152/tn02_cn025_lr001_bs128_e800/certify_sigma0.25_test",
        # "amlt/smoothing/tn02_cn025_lr001_bs128_e1600/resnet152/tn02_cn025_lr001_bs128_e1600/certify_sigma0.25_test",     

        # "amlt/smoothing/tn03_cn025_lr001_bs128_e100/resnet152/tn03_cn025_lr001_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tn03_cn025_lr001_bs128_e800/resnet152/tn03_cn025_lr001_bs128_e800/certify_sigma0.25_test",
        # "amlt/smoothing/tn03_cn025_lr001_bs128_e1600/resnet152/tn03_cn025_lr001_bs128_e1600/certify_sigma0.25_test",


        # expand fnoise
        # "amlt/smoothing/fexpn01a2_gn025_lr01_e100/resnet152/fexpn01a2_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn01a4_gn025_lr01_e100/resnet152/fexpn01a4_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn01a8_gn025_lr01_e100/resnet152/fexpn01a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn01a16_gn025_lr01_e100/resnet152/fexpn01a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn01a32_gn025_lr01_e100/resnet152/fexpn01a32_gn025_lr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/fexpn005a2_gn025_lr01_e100/resnet152/fexpn005a2_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn005a4_gn025_lr01_e100/resnet152/fexpn005a4_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn005a8_gn025_lr01_e100/resnet152/fexpn005a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn005a16_gn025_lr01_e100/resnet152/fexpn005a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn005a32_gn025_lr01_e100/resnet152/fexpn005a32_gn025_lr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/fexpn001a2_gn025_lr01_e100/resnet152/fexpn001a2_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn001a4_gn025_lr01_e100/resnet152/fexpn001a4_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn001a8_gn025_lr01_e100/resnet152/fexpn001a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn001a16_gn025_lr01_e100/resnet152/fexpn001a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn001a32_gn025_lr01_e100/resnet152/fexpn001a32_gn025_lr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/fexpn0005a2_gn025_lr01_e100/resnet152/fexpn0005a2_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn0005a4_gn025_lr01_e100/resnet152/fexpn0005a4_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn0005a8_gn025_lr01_e100/resnet152/fexpn0005a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn0005a16_gn025_lr01_e100/resnet152/fexpn0005a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn0005a32_gn025_lr01_e100/resnet152/fexpn0005a32_gn025_lr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/fexpn0001a2_gn025_lr01_e100/resnet152/fexpn0001a2_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn0001a4_gn025_lr01_e100/resnet152/fexpn0001a4_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn0001a8_gn025_lr01_e100/resnet152/fexpn0001a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn0001a16_gn025_lr01_e100/resnet152/fexpn0001a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/fexpn0001a32_gn025_lr01_e100/resnet152/fexpn0001a32_gn025_lr01_e100/certify_sigma0.25_train",


        # "amlt/smoothing/fexpn01a2_gn025_lr01_e100/resnet152/fexpn01a2_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn01a4_gn025_lr01_e100/resnet152/fexpn01a4_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn01a8_gn025_lr01_e100/resnet152/fexpn01a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn01a16_gn025_lr01_e100/resnet152/fexpn01a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn01a32_gn025_lr01_e100/resnet152/fexpn01a32_gn025_lr01_e100/certify_sigma0.25_test",

        # "amlt/smoothing/fexpn005a2_gn025_lr01_e100/resnet152/fexpn005a2_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn005a4_gn025_lr01_e100/resnet152/fexpn005a4_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn005a8_gn025_lr01_e100/resnet152/fexpn005a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn005a16_gn025_lr01_e100/resnet152/fexpn005a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn005a32_gn025_lr01_e100/resnet152/fexpn005a32_gn025_lr01_e100/certify_sigma0.25_test",

        # "amlt/smoothing/fexpn001a2_gn025_lr01_e100/resnet152/fexpn001a2_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn001a4_gn025_lr01_e100/resnet152/fexpn001a4_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn001a8_gn025_lr01_e100/resnet152/fexpn001a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn001a16_gn025_lr01_e100/resnet152/fexpn001a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn001a32_gn025_lr01_e100/resnet152/fexpn001a32_gn025_lr01_e100/certify_sigma0.25_test",

        # "amlt/smoothing/fexpn0005a2_gn025_lr01_e100/resnet152/fexpn0005a2_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn0005a4_gn025_lr01_e100/resnet152/fexpn0005a4_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn0005a8_gn025_lr01_e100/resnet152/fexpn0005a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn0005a16_gn025_lr01_e100/resnet152/fexpn0005a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn0005a32_gn025_lr01_e100/resnet152/fexpn0005a32_gn025_lr01_e100/certify_sigma0.25_test",  

        # "amlt/smoothing/fexpn0001a2_gn025_lr01_e100/resnet152/fexpn0001a2_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn0001a4_gn025_lr01_e100/resnet152/fexpn0001a4_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn0001a8_gn025_lr01_e100/resnet152/fexpn0001a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn0001a16_gn025_lr01_e100/resnet152/fexpn0001a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/fexpn0001a32_gn025_lr01_e100/resnet152/fexpn0001a32_gn025_lr01_e100/certify_sigma0.25_test",

        # "amlt/smoothing/rep_r152_n025_coslr01_bs128_e100_cp1/resnet152/rep_n025_coslr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/rep_r152_n025_coslr01_bs128_e100_cp1/resnet152/rep_n025_coslr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/tb_rep_n025_coslr01_bs128_e100/resnet152/rep_n025_coslr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/tb_rep_n025_coslr01_bs128_e100/resnet152/rep_n025_coslr01_bs128_e100/certify_sigma0.25_test",

        # nconv, fusion k noise in conv1 of r152, train
        # "amlt/smoothing/nconv_fn005a8_gn025_lr01_e100/resnet152/nconv_fn005a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/nconv_fn005a16_gn025_lr01_e100/resnet152/nconv_fn005a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/nconv_fn005a32_gn025_lr01_e100/resnet152/nconv_fn005a32_gn025_lr01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/nconv_fn01a8_gn025_lr01_e100/resnet152/nconv_fn01a8_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/nconv_fn01a8_gn025_lr01_e800/resnet152/nconv_fn01a8_gn025_lr01_e800/certify_sigma0.25_train",
        # "amlt/smoothing/nconv_fn01a8_gn025_lr01_e1600/resnet152/nconv_fn01a8_gn025_lr01_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/nconv_fn01a16_gn025_lr01_e100/resnet152/nconv_fn01a16_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/nconv_fn01a16_gn025_lr01_e800/resnet152/nconv_fn01a16_gn025_lr01_e800/certify_sigma0.25_train",
        # "amlt/smoothing/nconv_fn01a16_gn025_lr01_e1600/resnet152/nconv_fn01a16_gn025_lr01_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/nconv_fn01a32_gn025_lr01_e100/resnet152/nconv_fn01a32_gn025_lr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/nconv_fn01a32_gn025_lr01_e800/resnet152/nconv_fn01a32_gn025_lr01_e800/certify_sigma0.25_train",
        # "amlt/smoothing/nconv_fn01a32_gn025_lr01_e1600/resnet152/nconv_fn01a32_gn025_lr01_e1600/certify_sigma0.25_train",

        # nconv, fusion k noise in conv1 of r152, test
        # "amlt/smoothing/nconv_fn005a8_gn025_lr01_e100/resnet152/nconv_fn005a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/nconv_fn005a16_gn025_lr01_e100/resnet152/nconv_fn005a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/nconv_fn005a32_gn025_lr01_e100/resnet152/nconv_fn005a32_gn025_lr01_e100/certify_sigma0.25_test",

        # "amlt/smoothing/nconv_fn01a8_gn025_lr01_e100/resnet152/nconv_fn01a8_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/nconv_fn01a8_gn025_lr01_e800/resnet152/nconv_fn01a8_gn025_lr01_e800/certify_sigma0.25_test",
        # "amlt/smoothing/nconv_fn01a8_gn025_lr01_e1600/resnet152/nconv_fn01a8_gn025_lr01_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/nconv_fn01a16_gn025_lr01_e100/resnet152/nconv_fn01a16_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/nconv_fn01a16_gn025_lr01_e800/resnet152/nconv_fn01a16_gn025_lr01_e800/certify_sigma0.25_test",
        # "amlt/smoothing/nconv_fn01a16_gn025_lr01_e1600/resnet152/nconv_fn01a16_gn025_lr01_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/nconv_fn01a32_gn025_lr01_e100/resnet152/nconv_fn01a32_gn025_lr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/nconv_fn01a32_gn025_lr01_e800/resnet152/nconv_fn01a32_gn025_lr01_e800/certify_sigma0.25_test",
        # "amlt/smoothing/nconv_fn01a32_gn025_lr01_e1600/resnet152/nconv_fn01a32_gn025_lr01_e1600/certify_sigma0.25_test",

        # huggingface, vit_b_16 224 pretrained on clean in21k, finetuned on clean cf10, train on cf10 n025, adamw e100, resize to 224 then add noise
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_2_e100/hug/vit_224_in21k_ft_cf10_adamw1e_2_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_2_wd01_e100/hug/vit_224_in21k_ft_cf10_adamw1e_2_wd01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_3_e100/hug/vit_224_in21k_ft_cf10_adamw1e_3_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_3_wd01_e100/hug/vit_224_in21k_ft_cf10_adamw1e_3_wd01_e100/certify_sigma0.25_train", # best
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_4_e100/hug/vit_224_in21k_ft_cf10_adamw1e_4_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_4_wd01_e100/hug/vit_224_in21k_ft_cf10_adamw1e_4_wd01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_5_e100/hug/vit_224_in21k_ft_cf10_adamw1e_5_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_5_wd01_e100/hug/vit_224_in21k_ft_cf10_adamw1e_5_wd01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_6_e100/hug/vit_224_in21k_ft_cf10_adamw1e_6_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_6_wd01_e100/hug/vit_224_in21k_ft_cf10_adamw1e_6_wd01_e100/certify_sigma0.25_train",

        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_2_e100/hug/vit_224_in21k_ft_cf10_adamw1e_2_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_2_wd01_e100/hug/vit_224_in21k_ft_cf10_adamw1e_2_wd01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_3_e100/hug/vit_224_in21k_ft_cf10_adamw1e_3_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_3_wd01_e100/hug/vit_224_in21k_ft_cf10_adamw1e_3_wd01_e100/certify_sigma0.25_test", # best
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_4_e100/hug/vit_224_in21k_ft_cf10_adamw1e_4_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_4_wd01_e100/hug/vit_224_in21k_ft_cf10_adamw1e_4_wd01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_5_e100/hug/vit_224_in21k_ft_cf10_adamw1e_5_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_5_wd01_e100/hug/vit_224_in21k_ft_cf10_adamw1e_5_wd01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_6_e100/hug/vit_224_in21k_ft_cf10_adamw1e_6_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adamw1e_6_wd01_e100/hug/vit_224_in21k_ft_cf10_adamw1e_6_wd01_e100/certify_sigma0.25_test",

        # huggingface, vit_b_16 224 pretrained on clean in21k, finetuned on clean cf10, train on cf10 n025, adam, resize to 224 then add noise
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_2_e100/hug/vit_224_in21k_ft_cf10_adam1e_2_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_3_e100/hug/vit_224_in21k_ft_cf10_adam1e_3_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e100/hug/vit_224_in21k_ft_cf10_adam1e_4_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_5_e100/hug/vit_224_in21k_ft_cf10_adam1e_5_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_6_e100/hug/vit_224_in21k_ft_cf10_adam1e_6_e100/certify_sigma0.25_train",

        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_2_e100/hug/vit_224_in21k_ft_cf10_adam1e_2_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_3_e100/hug/vit_224_in21k_ft_cf10_adam1e_3_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e100/hug/vit_224_in21k_ft_cf10_adam1e_4_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_5_e100/hug/vit_224_in21k_ft_cf10_adam1e_5_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_6_e100/hug/vit_224_in21k_ft_cf10_adam1e_6_e100/certify_sigma0.25_test",

        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e5/hug/vit_224_in21k_ft_cf10_adam5e_5_e5/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e10/hug/vit_224_in21k_ft_cf10_adam5e_5_e10/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e50/hug/vit_224_in21k_ft_cf10_adam5e_5_e50/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e100/hug/vit_224_in21k_ft_cf10_adam5e_5_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e200/hug/vit_224_in21k_ft_cf10_adam5e_5_e200/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e400/hug/vit_224_in21k_ft_cf10_adam5e_5_e400/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e800/hug/vit_224_in21k_ft_cf10_adam5e_5_e800/certify_sigma0.25_train",

        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e5/hug/vit_224_in21k_ft_cf10_adam5e_5_e5/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e10/hug/vit_224_in21k_ft_cf10_adam5e_5_e10/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e50/hug/vit_224_in21k_ft_cf10_adam5e_5_e50/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e100/hug/vit_224_in21k_ft_cf10_adam5e_5_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e200/hug/vit_224_in21k_ft_cf10_adam5e_5_e200/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e400/hug/vit_224_in21k_ft_cf10_adam5e_5_e400/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam5e_5_e800/hug/vit_224_in21k_ft_cf10_adam5e_5_e800/certify_sigma0.25_test",

        # "amlt/smoothing/vit_224_in21k_ft_cf10_ctf/hug/vit_224_in21k_ft_cf10_ctf/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e1/hug/vit_224_in21k_ft_cf10_adam1e_4_e1/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e2/hug/vit_224_in21k_ft_cf10_adam1e_4_e2/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e3/hug/vit_224_in21k_ft_cf10_adam1e_4_e3/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e4/hug/vit_224_in21k_ft_cf10_adam1e_4_e4/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e5/hug/vit_224_in21k_ft_cf10_adam1e_4_e5/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e100/hug/vit_224_in21k_ft_cf10_adam1e_4_e100/certify_sigma0.25_train"

        # "amlt/smoothing/vit_224_in21k_ft_cf10_ctf/hug/vit_224_in21k_ft_cf10_ctf/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e1/hug/vit_224_in21k_ft_cf10_adam1e_4_e1/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e2/hug/vit_224_in21k_ft_cf10_adam1e_4_e2/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e3/hug/vit_224_in21k_ft_cf10_adam1e_4_e3/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e4/hug/vit_224_in21k_ft_cf10_adam1e_4_e4/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e5/hug/vit_224_in21k_ft_cf10_adam1e_4_e5/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e100/hug/vit_224_in21k_ft_cf10_adam1e_4_e100/certify_sigma0.25_test"

         # huggingface, vit_b_16 224 pretrained on clean in21k, finetuned on clean cf10, train on cf10 n025, adam, add noise on size 32, then resize to 224
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam1e_2_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam1e_2_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam1e_3_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam1e_3_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam1e_4_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam1e_4_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam5e_3_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam5e_3_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam5e_4_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam5e_4_e100/certify_sigma0.25_train",
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam5e_5_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam5e_5_e100/certify_sigma0.25_train",

        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam1e_2_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam1e_2_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam1e_3_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam1e_3_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam1e_4_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam1e_4_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam5e_3_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam5e_3_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam5e_4_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam5e_4_e100/certify_sigma0.25_test",
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam5e_5_e100/hug/vit_32n_rs224_in21k_ft_cf10_adam5e_5_e100/certify_sigma0.25_test",
       
        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam1e_3_e100_ctf_224n/hug/vit_32n_rs224_in21k_ft_cf10_adam1e_3_e100_ctf_224n/certify_sigma0.25_train",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e100_ctf_32n_rs224/hug/vit_224_in21k_ft_cf10_adam1e_4_e100_ctf_32n_rs224/certify_sigma0.25_train",

        # "amlt/smoothing/vit_32n_rs224_in21k_ft_cf10_adam1e_3_e100_ctf_224n/hug/vit_32n_rs224_in21k_ft_cf10_adam1e_3_e100_ctf_224n/certify_sigma0.25_test",
        # "amlt/smoothing/vit_224_in21k_ft_cf10_adam1e_4_e100_ctf_32n_rs224/hug/vit_224_in21k_ft_cf10_adam1e_4_e100_ctf_32n_rs224/certify_sigma0.25_test",


        # baseline resize, r152 train from scratch on CIFAR10 with noise 0.25, first resize to (64, 128, 224) then add noise
        # "amlt/smoothing/bl64_n025_coslr001_bs128_e100/resnet152/bl64_n025_coslr001_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/bl64_n025_coslr0001_bs128_e100/resnet152/bl64_n025_coslr0001_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/bl128_n025_coslr001_bs128_e100/resnet152/bl128_n025_coslr001_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/bl128_n025_coslr0001_bs128_e100/resnet152/bl128_n025_coslr0001_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/bl224_n025_coslr01_bs128_e100/resnet152/bl224_n025_coslr01_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/bl224_n025_coslr0001_bs128_e100/resnet152/bl224_n025_coslr0001_bs128_e100/certify_sigma0.25_train",

        # "amlt/smoothing/bl64_n025_coslr001_bs128_e100/resnet152/bl64_n025_coslr001_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/bl64_n025_coslr0001_bs128_e100/resnet152/bl64_n025_coslr0001_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/bl128_n025_coslr001_bs128_e100/resnet152/bl128_n025_coslr001_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/bl128_n025_coslr0001_bs128_e100/resnet152/bl128_n025_coslr0001_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/bl224_n025_coslr01_bs128_e100/resnet152/bl224_n025_coslr01_bs128_e100/certify_sigma0.25_test",
        # "amlt/smoothing/bl224_n025_coslr0001_bs128_e100/resnet152/bl224_n025_coslr0001_bs128_e100/certify_sigma0.25_test",


        # # diffusion model from carlini
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_ctf/diffusion/dif_vit_in21k_ft_cf10_ctf/certify_sigma0.25_train",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam1e_3_e100/diffusion/dif_vit_in21k_ft_cf10_adam1e_3_e100/certify_sigma0.25_train",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam1e_4_e100/diffusion/dif_vit_in21k_ft_cf10_adam1e_4_e100/certify_sigma0.25_train",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam5e_4_e100/diffusion/dif_vit_in21k_ft_cf10_adam5e_4_e100/certify_sigma0.25_train",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam5e_5_e100/diffusion/dif_vit_in21k_ft_cf10_adam5e_5_e100/certify_sigma0.25_train",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam1e_4_e10/diffusion/dif_vit_in21k_ft_cf10_adam1e_4_e10/certify_sigma0.25_train",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam1e_4_e50/diffusion/dif_vit_in21k_ft_cf10_adam1e_4_e50/certify_sigma0.25_train",

        # "amlt/smoothing/dif_vit_in21k_ft_cf10_ctf/diffusion/dif_vit_in21k_ft_cf10_ctf/certify_sigma0.25_test",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam1e_3_e100/diffusion/dif_vit_in21k_ft_cf10_adam1e_3_e100/certify_sigma0.25_test",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam1e_4_e100/diffusion/dif_vit_in21k_ft_cf10_adam1e_4_e100/certify_sigma0.25_test",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam5e_4_e100/diffusion/dif_vit_in21k_ft_cf10_adam5e_4_e100/certify_sigma0.25_test",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam5e_5_e100/diffusion/dif_vit_in21k_ft_cf10_adam5e_5_e100/certify_sigma0.25_test",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam1e_4_e10/diffusion/dif_vit_in21k_ft_cf10_adam1e_4_e10/certify_sigma0.25_test",
        # "amlt/smoothing/dif_vit_in21k_ft_cf10_adam1e_4_e50/diffusion/dif_vit_in21k_ft_cf10_adam1e_4_e50/certify_sigma0.25_test",

        # certify CIFAR10 full test set
        # "amlt/smoothing/r152_imgn32n025e100_coslr001_bs128_n025_e100_full_ctf/resnet152/r152_imgn32n025e100_coslr001_bs128_n025_e100_full_ctf/certify_sigma0.25_test", 

        # imagenet1k, size224, train from scratch with n025
        # "amlt/smoothing/r152_n025_lr1_bs64_e100/imagenet/r152_n025_lr1_bs64_e100/certify_sigma0.25_train",
        # "amlt/smoothing/r152_n025_lr1_bs128_e100/imagenet/r152_n025_lr1_bs128_e100/certify_sigma0.25_train",
        # "amlt/smoothing/r152_n025_lr1_bs256_e100/imagenet/r152_n025_lr1_bs256_e100/certify_sigma0.25_train",
        # "amlt/smoothing/r152_n025_lr01_bs256_e100/imagenet/r152_n025_lr01_bs256_e100/certify_sigma0.25_train",
        # "amlt/smoothing/r152_n025_lr001_bs256_e100/imagenet/r152_n025_lr001_bs256_e100/certify_sigma0.25_train",
        # "amlt/smoothing/r152_n025_lr0001_bs256_e100/imagenet/r152_n025_lr0001_bs256_e100/certify_sigma0.25_train",
        # "amlt/smoothing/r152_n025_lr00001_bs256_e100/imagenet/r152_n025_lr00001_bs256_e100/certify_sigma0.25_train",

        "amlt/smoothing/r152_n025_lr1_bs64_e100/imagenet/r152_n025_lr1_bs64_e100/certify_sigma0.25_test",
        "amlt/smoothing/r152_n025_lr1_bs128_e100/imagenet/r152_n025_lr1_bs128_e100/certify_sigma0.25_test",
        "amlt/smoothing/r152_n025_lr1_bs256_e100/imagenet/r152_n025_lr1_bs256_e100/certify_sigma0.25_test",
        "amlt/smoothing/r152_n025_lr01_bs256_e100/imagenet/r152_n025_lr01_bs256_e100/certify_sigma0.25_test",
        "amlt/smoothing/r152_n025_lr001_bs256_e100/imagenet/r152_n025_lr001_bs256_e100/certify_sigma0.25_test",
        "amlt/smoothing/r152_n025_lr0001_bs256_e100/imagenet/r152_n025_lr0001_bs256_e100/certify_sigma0.25_test",
        "amlt/smoothing/r152_n025_lr00001_bs256_e100/imagenet/r152_n025_lr00001_bs256_e100/certify_sigma0.25_test",

    ]

    plot_certified_accuracy(
        "../amlt/smoothing/analysis/plots/in_r152_n025_e100_test", "in_r152_n025_e100_test", 1.0, [
            Line(ApproximateAccuracy(os.path.join('../', ctf_file)), ctf_file.split('/')[2]) for ctf_file in ctf_files
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
