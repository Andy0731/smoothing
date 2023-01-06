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
    plot_certified_accuracy(ctf_filename, exp, 4.0, lines)

if __name__ == "__main__":
    ctf_files = [

        # "amlt/smoothing/r110_n025_coslr01_bs16_e200_cert_train/resnet110/n025_coslr01_bs16_e200_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs16_e400_cert_train/resnet110/n025_coslr01_bs16_e400_cert_train/certify_sigma0.25",
        # "amlt/smoothing/r110_n025_coslr01_bs16_e800_cert_train/resnet110/n025_coslr01_bs16_e800_cert_train/certify_sigma0.25",

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

        # "amlt/smoothing/r152_n025_lr01_bs128_e100/resnet152/n025_lr01_bs128_e100/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_lr01_bs128_e200/resnet152/n025_lr01_bs128_e200/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_lr01_bs128_e400/resnet152/n025_lr01_bs128_e400/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_lr01_bs128_e800/resnet152/n025_lr01_bs128_e800/certify_sigma0.25",
        # "amlt/smoothing/r152_n025_coslr01_bs128_e1600_m2/resnet152/n025_coslr01_bs128_e1600/certify_sigma0.25_train",
        # "amlt/smoothing/r152_n025_coslr01_bs128_e1600_m2/resnet152/n025_coslr01_bs128_e1600/certify_sigma0.25_test",
        # "amlt/smoothing/r152_n025_coslr01_bs128_e12800_m8/resnet152/n025_coslr01_bs128_e12800/certify_sigma0.25_train",
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
        # r110 n0 bs256 e100 train
        # "amlt/smoothing/imgn32_r110_n0_coslr001_bs256_e100/imagenet32/r110_n0_coslr001_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr01_bs256_e100/imagenet32/r110_n0_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e100/imagenet32/r110_n0_coslr1_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr10_bs256_e100/imagenet32/r110_n0_coslr10_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr100_bs256_e100/imagenet32/r110_n0_coslr100_e100/certify_sigma0.25_train",

        # # r110 n0 bs256 e100 test
        # "amlt/smoothing/imgn32_r110_n0_coslr001_bs256_e100/imagenet32/r110_n0_coslr001_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n0_coslr01_bs256_e100/imagenet32/r110_n0_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e100/imagenet32/r110_n0_coslr1_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n0_coslr10_bs256_e100/imagenet32/r110_n0_coslr10_e100/certify_sigma0.25_test",
        # "amlt/smoothing/imgn32_r110_n0_coslr100_bs256_e100/imagenet32/r110_n0_coslr100_e100/certify_sigma0.25_test",

        # # r110 n0 bs256 coslr1 train
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e100/imagenet32/r110_n0_coslr1_e100/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e200/imagenet32/r110_n0_coslr1_e200/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e400/imagenet32/r110_n0_coslr1_e400/certify_sigma0.25_train",
        # "amlt/smoothing/imgn32_r110_n0_coslr1_bs256_e800/imagenet32/r110_n0_coslr1_e800/certify_sigma0.25_train",    

        # # r110 n0 bs256 coslr1 test
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

        # # r152 pretrain on imagenet32 with noise_sd 0.25, vary epochs, train
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e100_m8/imagenet32/r152_n025_coslr1_e100/certify_sigma0.25_train",
        # # "amlt/smoothing/pt_imgn32_r152_n025_coslr01_e100_m8/imagenet32/r152_n025_coslr01_e100/certify_sigma0.25_train",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e200/imagenet32/r152_n025_coslr1_e200/certify_sigma0.25_train",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e400_m16/imagenet32/r152_n025_coslr1_e400/certify_sigma0.25_train",

        # # r152 pretrain on imagenet32 with noise_sd 0.25, vary epochs, test
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e100_m8/imagenet32/r152_n025_coslr1_e100/certify_sigma0.25_test",
        # # "amlt/smoothing/pt_imgn32_r152_n025_coslr01_e100_m8/imagenet32/r152_n025_coslr01_e100/certify_sigma0.25_test",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e200/imagenet32/r152_n025_coslr1_e200/certify_sigma0.25_test",
        # "amlt/smoothing/pt_imgn32_r152_n025_coslr1_e400_m16/imagenet32/r152_n025_coslr1_e400/certify_sigma0.25_test",

        # # r152 pretrain on imagenet32 with noise_sd 0.25, vary epochs, finetune on cifar10 coslr001 bs128 n025 e100, train
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e200_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e200_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        # "amlt/smoothing/ft_r152_imgn32n025e400_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e400_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # # r152 pretrain on imagenet32 with noise_sd 0.25, vary epochs, finetune on cifar10 coslr001 bs128 n025 e100, test
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e200_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e200_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_imgn32n025e400_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e400_coslr001_bs128_n025_e100/certify_sigma0.25_test",

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

        # r300 vs r152, pretrain on imagenet32
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

        # r300 vs r152, imgn32 vs ti500k, pretrain + finetune, train
        "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",
        "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",     
        "amlt/smoothing/ft_r300_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r300_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_train",

        # # r300 vs r152, imgn32 vs ti500k, pretrain + finetune, test
        # "amlt/smoothing/ft_r152_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r152_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",
        # "amlt/smoothing/ft_r152_ti500kn025lr01e100_coslr001_bs128_n025_e100_P100/finetune/r152_ti500kn025lr01e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",     
        # "amlt/smoothing/ft_r300_imgn32n025e100_coslr001_bs128_n025_e100/finetune/r300_imgn32n025e100_coslr001_bs128_n025_e100/certify_sigma0.25_test",

    ]

    plot_certified_accuracy(
        "../amlt/smoothing/analysis/plots/ft_r300_vs_r152_imgn32_vs_ti500k_train", "ft_r300_vs_r152_imgn32_vs_ti500k_train", 1.0, [
            Line(ApproximateAccuracy(os.path.join('../', ctf_file)), ctf_file.split('/')[2].replace('ft_','').replace('coslr001_bs128_n025_e100','').replace('_P100','')) for ctf_file in ctf_files
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
