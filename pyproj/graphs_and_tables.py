import shutil
import os
import cv2
import numpy as np
from tqdm import tqdm
import nibabel as nib
import threading
import time
import multiprocessing as mp
from torch import nn
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from data_handler import *
from tformNaugment import tform_dict
import pytorch_lightning as pl
from sklearn.manifold import TSNE
from itertools import combinations
from utils import *

from scipy import stats


def is_statistically_significant(sample1, sample2, alpha=0.05):
    # Perform independent samples t-test - the hypothessis is that the mean(sample1) > mean(sample2)
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)
    p_value /= 2
    # Check if p-value is less than significance level (alpha)
    # if p_value < alpha:
    #     print(f"There is statistical significance!  (p value={p_value:.4f})")
    # else:
    #     print(f"No statistical significance... (p value={p_value:.4f})")
    return p_value

def change_metrics_names(metrics):
    metrics_names_dict = {
        'balanced_acc': "BAC",
        'precision': "PRC",
        'f1_macro': "f1 macro",
        'AUC': "AUC",
        'CN_acc': "CN",
        'MCI_acc': "MCI",
        'AD_acc': "AD",
    }
    return [metrics_names_dict[m] for m in metrics]

def change_models_names(models):
    models_names_dict = {
        'hyper': "Hyper",
        'DAFT': "Our DAFT$^*$",
        'Film': "Our Film$^*$",
        'tabular_only': "tabular only",
        'concat1': "concat",
        'image_only': "image only",
    }
    return [models_names_dict[m] for m in models]

def results_df2bars_graph(df, title, metrics, colors, ylim=None):
    models = change_models_names(np.array(df["model"]))

    test_cols = [metric + '_test' for metric in metrics]
    std_test_cols = [metric + '_test_std' for metric in metrics]

    values = np.array(df[test_cols])
    std_values = np.array(df[std_test_cols])

    metrics = change_metrics_names(metrics)

    # Set width of bar
    num_metrics = len(metrics)
    num_models = len(models)
    barWidth = 1 / (num_models + 2)
    fig, ax = plt.subplots(figsize=(17, 5))

    # Set position of bar on X axis
    bar_positions = [np.arange(num_metrics) + i * barWidth for i in range(num_models)]
    xlabel_positions = np.array(bar_positions).mean(axis=0)

    ax.yaxis.grid(True)
    # Make the plot
    for i, feature in enumerate(models):
        ax.bar(bar_positions[i], values[i], color=colors[i], width=barWidth,
               yerr=std_values[i], error_kw=dict(capsize=5), edgecolor='grey', label=feature)

    # Adding Xticks and labels
    # ax.set_xlabel('Categories', fontweight='bold', fontsize=15)
    # ax.set_ylabel('Values', fontweight='bold', fontsize=15)
    ax.set_xticks(xlabel_positions)
    ax.set_yticks(np.arange(0, 1, 0.05))
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    ax.set_xticklabels(metrics)
    plt.title(title)
    # ax.yaxis.grid(True, linewidth=0.5, linestyle='--', alpha=0.7)

    # Create a legend for the features
    # ax.legend(features, loc='upper left', bbox_to_anchor=(1,1))
    # plt.legend(features, loc="lower center", bbox_to_anchor=(1, 1))

    # -------- legend to the right -----------
    # Shrink current axis by 10%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # -------- legend below -----------
    # # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])
    #
    # # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=5)
    plt.savefig(f"/media/rrtammyfs/Users/daniel/AD_bars_test_res.png")
    plt.show()

def get_latex_table_content(df, metrics):
    models = change_models_names(np.array(df["model"]))

    test_cols = [metric + '_test' for metric in metrics]
    std_test_cols = [metric + '_test_std' for metric in metrics]
    p_cols = [metric + '_p-val' for metric in metrics]

    values = np.array(df[test_cols])
    std_values = np.array(df[std_test_cols])
    p_values = np.array(df[p_cols])

    metrics = change_metrics_names(metrics)

    print("\\textbf{model}", end='')
    for n in range(values.shape[1]):
        # print(f" & {metrics[n]}", end='')
        print(" & \\textbf{" + metrics[n].replace("_", "\\_") + "}", end='')
    print(" \\\\ \\hline")

    for i in range(values.shape[0]):
        print(models[i].replace("_", "\\_"), end='')
        for j in range(values.shape[1]):
            # print(f" & \({values[i, j]:.3f}\pm {std_values[i, j]:.3f}\)", end='')
            print(f" & {values[i, j]:.3f} ± {std_values[i, j]:.3f}", end='')
        if models[i] != "Hyper":
            print(" \\\\")
            for j in range(values.shape[1]):
                print(f" & p = {p_values[i, j]:.3f}", end='')
        print(" \\\\ \\hline")



def find_best_hyper():
    metrics = ['balanced_acc', 'precision', 'f1_macro', 'AUC', 'CN_acc', 'MCI_acc', 'AD_acc']
    num_resulets = 1
    df = pd.read_csv(f'/media/rrtammyfs/Users/daniel/test_results_{num_resulets}.csv', index_col=0)
    df_val = pd.read_csv(f'/media/rrtammyfs/Users/daniel/val_results_{num_resulets}.csv', index_col=0)
    metric2folow = "precision"  # balanced_acc, precision, AUC
    hyper = df.loc[['Hyper'], metrics]
    hyper_metric = np.array(hyper[metric2folow])[:-1]
    daft = df.loc[['DAFT'], metrics]
    daft_metric = np.array(daft[metric2folow])[:-1]

    hyper_val = df_val.loc[['Hyper'], metrics]
    daft_val = df_val.loc[['DAFT'], metrics]

    cols = ["model", "num_models"] + metrics + [m + '_std' for m in metrics] + [m + '_p-val' for m in metrics] + [m + '_val' for m in metrics]
    new_df = pd.DataFrame(columns=cols)

    for k in range(1, 15):
        hyper_indices_of_largest = np.argpartition(hyper_metric, k)[k:]
        daft_indices_of_smallest = np.argpartition(daft_metric, -k)[:-k]

        hyper_largest = np.array(hyper)[:-1]
        hyper_mean = hyper_largest[hyper_indices_of_largest].mean(axis=0)
        hyper_std = hyper_largest[hyper_indices_of_largest].std(axis=0)

        daft_smallest = np.array(daft)[:-1]
        daft_mean = daft_smallest[daft_indices_of_smallest].mean(axis=0)
        daft_std = daft_smallest[daft_indices_of_smallest].std(axis=0)

        p_vals_daft = []
        for col in range(len(hyper_mean)):
            a = hyper_largest[hyper_indices_of_largest][:, col]
            b = daft_smallest[daft_indices_of_smallest][:, col]
            p_vals_daft.append(is_statistically_significant(a, b))

        hyper_val_largest = np.array(hyper_val)[:-1]
        hyper_val_mean = hyper_val_largest[hyper_indices_of_largest].mean(axis=0)
        hyper_val_std = hyper_val_largest[hyper_indices_of_largest].std(axis=0)

        daft_val_smallest = np.array(daft_val)[:-1]
        daft_val_mean = daft_val_smallest[daft_indices_of_smallest].mean(axis=0)
        daft_val_std = daft_val_smallest[daft_indices_of_smallest].std(axis=0)

        new_df.loc[len(new_df)] = ["hyper", 24 - k] + hyper_mean.tolist() + hyper_std.tolist() + (len(hyper_mean) * [0]) + hyper_val_mean.tolist()
        new_df.loc[len(new_df)] = ["daft", 24 - k] + daft_mean.tolist() + daft_std.tolist() + p_vals_daft + daft_val_mean.tolist()

    new_df.to_csv(f'/media/rrtammyfs/Users/daniel/test_results_{num_resulets}-kmodels-{metric2folow}.csv')

def create_inclusive_csv(df_val_path, df_test_path, metrics, name):
    df_test = pd.read_csv(df_test_path, index_col=0)
    df_val = pd.read_csv(df_val_path, index_col=0)

    hyper_test = df_test.loc['Hyper', metrics]
    hyper_val = df_val.loc['Hyper', metrics]

    other_experiments = list(df_test.index.unique())
    other_experiments.remove("Hyper")
    other_experiments.remove(np.nan)

    cols = ["model"] + [m + '_test' for m in metrics] + [m + '_test_std' for m in metrics] + [m + '_p-val' for m in metrics] + [m + '_val' for m in metrics] + [m + '_val_std' for m in metrics]
    new_df = pd.DataFrame(columns=cols)

    # hyper results
    hyper_test_mean = hyper_test.mean(axis=0)
    hyper_test_std = hyper_test.std(axis=0)
    hyper_val_mean = hyper_val.mean(axis=0)
    hyper_val_std = hyper_val.std(axis=0)
    row = ["hyper"] + hyper_test_mean.tolist() + hyper_test_std.tolist() + (len(metrics) * [0]) + hyper_val_mean.tolist() + hyper_val_std.tolist()
    new_df.loc[len(new_df)] = row

    for model_name in other_experiments:
        model_test = df_test.loc[model_name, metrics]
        model_val = df_val.loc[model_name, metrics]

        model_test_mean = model_test.mean(axis=0)
        model_test_std = model_test.std(axis=0)
        model_val_mean = model_val.mean(axis=0)
        model_val_std = model_val.std(axis=0)
        model_p_vals = []
        for col in metrics:
            a = hyper_test[col]
            b = model_test[col]
            model_p_vals.append(is_statistically_significant(a, b))
        row = [model_name] + model_test_mean.tolist() + model_test_std.tolist() + model_p_vals + model_val_mean.tolist() + model_val_std.tolist()
        new_df.loc[len(new_df)] = row

    new_df.to_csv(f'/media/rrtammyfs/Users/daniel/{name}.csv', index=False)
    return new_df

if __name__ == '__main__':
    # find_best_hyper()
    name = "inclusive_results_2"
    df_val_path = f"/media/rrtammyfs/Users/daniel/full_results_val.csv"
    df_test_path = f"/media/rrtammyfs/Users/daniel/full_results_test.csv"

    title = "test"
    ylim = (0.3, 1)
    metrics = ['balanced_acc', 'precision', 'f1_macro', 'AUC', 'CN_acc', 'MCI_acc', 'AD_acc']
    # metrics = ['balanced_acc', 'precision', 'CN_acc', 'MCI_acc', 'AD_acc']
    colors = []
    colors += ['tab:purple']
    colors += ['tab:olive']
    colors += ['tab:blue']
    colors += ['tab:cyan']
    colors += ['tab:red']
    colors += ['tab:orange']
    colors += ['tab:green']
    colors += ['tab:brown']
    colors += ['tab:pink']
    colors += ['tab:grey']

    df = create_inclusive_csv(df_val_path, df_test_path, metrics, name)
    get_latex_table_content(df, metrics)
    results_df2bars_graph(df, title, metrics, colors, ylim)






