import os
import numpy as np
from tqdm import tqdm
from scipy import stats
import json
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Initialize the API
api = wandb.Api()


def get_accNprecision(run, eval_type):
    artif = api.artifact(f'HyperNetworks_final/run-{run.id}-{eval_type}_confmatraw_confmat:latest')
    path = artif.download()
    with open(path + f'/{eval_type}_confmat/raw_confmat.table.json', 'rb') as f:
        confusion_matrix = json.load(f)["data"]
    confusion_matrix = np.array(confusion_matrix)
    precision = (np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)).mean()
    balanced_acc = (np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)).mean()  # == recall
    return balanced_acc, precision

def get_precision_from_confmat(run, eval_type, project_name="HyperNetworks_final_splitseed"):
    artif = api.artifact(f'{project_name}/run-{run.id}-{eval_type}_confmatraw_confmat:latest')
    path = artif.download()
    with open(path + f'/{eval_type}_confmat/raw_confmat.table.json', 'rb') as f:
        confusion_matrix = json.load(f)["data"]
    confusion_matrix = np.array(confusion_matrix)
    precision = np.nan_to_num((np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)), nan=0).mean()
    return precision

def get_confusion_matrix(run, eval_type):
    artif = api.artifact(f'{run.project}/run-{run.id}-{eval_type}_confmatraw_confmat:latest')
    path = artif.download()
    with open(path + f'/{eval_type}_confmat/raw_confmat.table.json', 'rb') as f:
        confusion_matrix = json.load(f)["data"]
    confusion_matrix = np.array(confusion_matrix)
    return confusion_matrix

def get_runs_test_metrics(runs, eval_type, metric):
    # print(f"getting runs' metrics from wandb - {metric}:")
    # Iterate over the runs and access their results
    metrics = []
    for run in runs:
        # results = run.history()
        if metric == "balanced_acc":
            value = run.summary[f'{eval_type}/best_balanced_acc']
        elif metric == "AUC":
            value = run.summary[f'{eval_type}/AUC_at_best_point']
        elif metric == "f1_macro":
            value = run.summary[f'{eval_type}/f1_macro_at_best_point']
        elif metric == "f1_micro":
            value = run.summary[f'{eval_type}/f1_micro_at_best_point']
        elif metric == "precision":
            value = run.summary[f'{eval_type}/precision_at_best_point']
            if value == "NaN":
                value = get_precision_from_confmat(run, eval_type)
        elif metric == "CN_acc":
            value = run.summary[f'{eval_type}/CN_acc_at_best_point']
        elif metric == "MCI_acc":
            value = run.summary[f'{eval_type}/MCI_acc_at_best_point']
        elif metric == "AD_acc":
            value = run.summary[f'{eval_type}/AD_acc_at_best_point']
        elif metric == "confusion_matrix":
            value = get_confusion_matrix(run, eval_type)
        else:
            # balanced_acc, precision = get_accNprecision(run, eval_type)
            raise ValueError("this metric is not logged")
        metrics.append(value)
    return metrics


def is_statistically_significant(sample1, sample2, alpha=0.05):
    # Perform independent samples t-test - the hypothessis is that the mean(sample1) > mean(sample2)
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)
    p_value /= 2
    # Check if p-value is less than significance level (alpha)
    if p_value < alpha:
        print(f"There is statistical significance!  (p value={p_value:.4f})")
    else:
        print(f"No statistical significance... (p value={p_value:.4f})")
    return p_value


def check_statistics(project_name, experiment_name_a, experiment_name_b, eval_type, metric_name):
    # Fetch the runs that match the project and run name
    # runs_a = api.runs(project_name, filters={"config.experiment_name": {"$regex": experiment_name_a}})
    if type(experiment_name_a) == list:
        experiment_name_a = "|".join(experiment_name_a)
    if type(experiment_name_b) == list:
        experiment_name_b = "|".join(experiment_name_b)
    runs_a = api.runs(project_name, filters={"config.experiment_name": {"$regex": experiment_name_a}})
    runs_b = api.runs(project_name, filters={"config.experiment_name": {"$regex": experiment_name_b}})

    if eval_type == "test":
        runs_a = [run for run in runs_a if "Eval" in run.name]
        runs_b = [run for run in runs_b if "Eval" in run.name]
    elif eval_type == "val":
        runs_a = [run for run in runs_a if not ("Eval" in run.name)]
        runs_b = [run for run in runs_b if not ("Eval" in run.name)]
    else:
        raise ValueError("val or test only!!")

    print("runs a:")
    for run in runs_a:
        print(run.name)
    print("runs b:")
    for run in runs_b:
        print(run.name)

    metrics_a = get_runs_test_metrics(runs=runs_a, eval_type=eval_type, metric=metric_name)
    metrics_b = get_runs_test_metrics(runs=runs_b, eval_type=eval_type, metric=metric_name)

    # Check for statistical significance
    print("\n--------- experiments statistics ---------")
    print(f"metric name: {metric_name}")
    print(f"{experiment_name_a.split('|')[0]}:  {np.mean(metrics_a):.4f} +- {np.std(metrics_a):.4f}")
    print(f"{experiment_name_b.split('|')[0]}:  {np.mean(metrics_b):.4f} +- {np.std(metrics_b):.4f}")

    p_value = is_statistically_significant(metrics_a, metrics_b)
    return f"{np.mean(metrics_a):.4f} +- {np.std(metrics_a):.4f}", f"{np.mean(metrics_b):.4f} +- {np.std(metrics_b):.4f}", p_value


def get_experiment_metrics(project_name, experiment_name, eval_type, metrics_names, regex=False):
    if regex:
        if type(experiment_name) == list:
            experiment_name = "|".join(experiment_name)
        runs = api.runs(project_name, filters={"config.experiment_name": {"$regex": experiment_name}})
    else:
        runs = api.runs(project_name, filters={"config.experiment_name": experiment_name})

    if eval_type == "test":
        runs = [run for run in runs if "Eval" in run.name]
    elif eval_type == "val":
        runs = [run for run in runs if not ("Eval" in run.name)]
    else:
        raise ValueError("val or test only!!")

    print("\nruns:")
    for run in runs:
        print(run.name)
    print("")

    metrics_dict = {"experiment_name": experiment_name.split("|")[0].replace("v1-seed0", "")}
    for metric in metrics_names:
        m = get_runs_test_metrics(runs=runs, eval_type=eval_type, metric=metric)
        metrics_dict[metric] = f"{np.mean(m):.4f}"
        # metrics_dict[metric + "_std"] = f"{np.std(m):.4f}"
        # metrics_dict[metric + "_mean_N_std"] = f"{np.mean(m):.4f} +- {np.std(m):.4f}"

    return metrics_dict

def get_experiment_confmat(project_name, experiment_name, eval_type, save_path, regex=False):
    if regex:
        if type(experiment_name) == list:
            experiment_name = "|".join(experiment_name)
        runs = api.runs(project_name, filters={"config.experiment_name": {"$regex": experiment_name}})
    else:
        runs = api.runs(project_name, filters={"config.experiment_name": experiment_name})

    if eval_type == "test":
        runs = [run for run in runs if "Eval" in run.name]
    elif eval_type == "val":
        runs = [run for run in runs if not ("Eval" in run.name)]
    else:
        raise ValueError("val or test only!!")

    print("\nruns:")
    for run in runs:
        print(run.name)
    print("")

    confusion_matrix_lst = get_runs_test_metrics(runs=runs, eval_type=eval_type, metric="confusion_matrix")

    # normalize
    for confusion_matrix in confusion_matrix_lst:
        for i in range(confusion_matrix.shape[0]):
            confusion_matrix[i] /= confusion_matrix[i].sum()

    labels_y = ["CN", 'MCI', "AD"]
    labels_x = ["CN", 'MCI', "AD"]
    confusion_matrix_mean = np.mean(confusion_matrix_lst, axis=0).round(4)
    confusion_matrix_std = np.std(confusion_matrix_lst, axis=0).round(4)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot average confusion matrix
    heatmap = ax.imshow(confusion_matrix_mean, cmap='Reds')

    # Overlay error matrix
    for i in range(confusion_matrix_mean.shape[0]):
        for j in range(confusion_matrix_mean.shape[1]):
            color = "black"
            if confusion_matrix_mean[i, j] > confusion_matrix_mean.max()/2:
                color = "white"
            ax.text(j, i, f'{confusion_matrix_mean[i, j]:.3f}\n±{confusion_matrix_std[i, j]:.3f}', ha='center', va='center', color=color)

    # Add colorbar
    cbar = plt.colorbar(heatmap)


    # ax.set_title('Confusion Matrix (Mean ± Std)', loc='center')
    # ax.set_title(f'                                       prediction                    ({os.path.basename(save_path).split(".")[0]})', loc='center')
    # ax.set_title(f'prediction', loc='center')
    # ax.set_title(f'Confusion Matrix (Mean ± Std) - {os.path.basename(save_path).split(".")[0]}', loc='center')
    ax.text(0.5, 1.07, 'prediction', transform=ax.transAxes, ha='center')
    # ax.text(0.5, 1.11, f'Confusion Matrix (Mean ± Std) - {os.path.basename(save_path).split(".")[0]}', transform=ax.transAxes, ha='center', fontsize=11)
    ax.text(0.5, 1.11, f'Confusion Matrix - {os.path.basename(save_path).split(".")[0].split("_")[0]}', transform=ax.transAxes, ha='center', fontsize=11)

    # Add secondary title

    plt.ylabel('Ground Truth')
    plt.xticks(np.arange(confusion_matrix_mean.shape[1]), labels_x)
    plt.yticks(np.arange(confusion_matrix_mean.shape[0]), labels_y)
    fig.axes[0].xaxis.tick_top()

    plt.savefig(save_path, bbox_inches='tight')
    # Show the plot
    plt.show()


def arange_results_csv(df):
    new_col_names, features_set_col = change_columns_names(df)
    columns = list(df.columns)
    columns.remove("experiment_name")
    new_df = pd.DataFrame(index=new_col_names, data=np.concatenate([np.array(features_set_col)[None].T, df[columns].to_numpy()], axis=1), columns=["features_set"] + columns)
    # for col in columns:
    #     new_df[col] = new_df[col].str.split(' ').str.get(0)

    return new_df

def change_columns_names(df):
    # new_col_names = list(df.experiment_name.str.split('_').str.get(0) + '-' + df.experiment_name.str.split('-').str.get(-1))
    new_col_names = list(df.experiment_name.str.split('_').str.get(0))
    for i, name in enumerate(new_col_names):
        name = name.replace("baseline-resnet", "image_only")
        name = name.replace("baseline-tabular", "tabular_only")
        name = name.replace("baseline-concat1", "concat1")
        name = name.replace("TabularAsHyper", "Hyper")
        name = name.replace("fs12", "9_features")
        name = name.replace("fs8", "4_features")
        name = name.replace("fs0", "2_features")
        name = name.replace("fs5", "1_features")
        new_col_names[i] = name
    return new_col_names, list(df.experiment_name.str.split('-').str.get(-1))



if __name__ == "__main__":
    project_name = "HyperNetworks_final_splitseed"

    metrics_names = ["balanced_acc", "precision", "f1_macro", "AUC", "CN_acc", "MCI_acc", "AD_acc"]
    experiments_names_dict = {
        "hyper_fs12": [
            "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v1-seed0-fs12",
            "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v2-seed0-fs12",
            # "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v3-seed0-fs12",
            "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v1-seed1-fs12",
            "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v2-seed1-fs12",
            # "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v3-seed1-fs12",
            # "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v1-seed2-fs12",
            "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v2-seed2-fs12",
            # "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v3-seed2-fs12",
            "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v1-seed3-fs12",
            "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v2-seed3-fs12",
            # "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v1-seed4-fs12",
            # "TabularAsHyper_R_R_R_FFT_FF_embd_trainedTabular_cw085_v2-seed4-fs12",

            # "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v1-seed0-fs12",
            # "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v2-seed0-fs12",
            # # "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v1-seed1-fs12",
            # "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v2-seed1-fs12",
            # "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v1-seed2-fs12",
            # "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v2-seed2-fs12",
        ],
        "hyper_fs8": [
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v1-seed0-fs8",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v2-seed0-fs8",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v1-seed1-fs8",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v2-seed1-fs8",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v1-seed2-fs8",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v2-seed2-fs8",
        ],
        "hyper_fs0": [
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v1-seed0-fs0",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v2-seed0-fs0",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v1-seed1-fs0",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v2-seed1-fs0",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v1-seed2-fs0",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1_v2-seed2-fs0",
        ],
        "hyper_fs5": [
            # "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1v1-seed0-fs5",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1v2-seed0-fs5",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1v1-seed1-fs5",
            "TabularAsHyper_R_R_R_FFT_FF_embd8_cw1v2-seed1-fs5",
        ],

        "DAFT_fs12": [
            "DAFT_BalancCw_v1-seed0-fs12",
            "DAFT_BalancCw_v2-seed0-fs12",
            # "DAFT_BalancCw_v1-seed1-fs12",
            "DAFT_BalancCw_v2-seed1-fs12",
            "DAFT_BalancCw_v1-seed2-fs12",
            "DAFT_BalancCw_v2-seed2-fs12",
            "DAFT_BalancCw_v1-seed3-fs12",
            "DAFT_BalancCw_v2-seed3-fs12",
        ],
        "DAFT_fs8": [
            "DAFT_BalancCw_v1-seed0-fs8",
            "DAFT_BalancCw_v2-seed0-fs8",
            "DAFT_BalancCw_v1-seed1-fs8",
            "DAFT_BalancCw_v2-seed1-fs8",
            "DAFT_BalancCw_v1-seed2-fs8",
            "DAFT_BalancCw_v2-seed2-fs8",
        ],
        "DAFT_fs0": [
            "DAFT_BalancCw_v1-seed0-fs0",
            "DAFT_BalancCw_v2-seed0-fs0",
            "DAFT_BalancCw_v1-seed1-fs0",
            "DAFT_BalancCw_v2-seed1-fs0",
            "DAFT_BalancCw_v1-seed2-fs0",
            "DAFT_BalancCw_v2-seed2-fs0",
        ],
        "DAFT_fs5": [
            "DAFT_BalancCw_v1-seed0-fs5",
            "DAFT_BalancCw_v2-seed0-fs5",
            "DAFT_BalancCw_v1-seed1-fs5",
            "DAFT_BalancCw_v2-seed1-fs5",
        ],

        "Film_fs12": [
            "Film_cw_v1-seed0-fs12",
            "Film_cw_v2-seed0-fs12",
            "Film_cw_v1-seed1-fs12",
            "Film_cw_v2-seed1-fs12",
            "Film_cw_v1-seed2-fs12",
            # "Film_cw_v2-seed2-fs12",
        ],
        "Film_fs8": [
            "Film_cw_v1-seed0-fs8",
            "Film_cw_v2-seed0-fs8",
            "Film_cw_v1-seed1-fs8",
            "Film_cw_v2-seed1-fs8",
            "Film_cw_v1-seed2-fs8",
            "Film_cw_v2-seed2-fs8",
        ],
        "Film_fs0": [
            "Film_cw_v1-seed0-fs0",
            "Film_cw_v2-seed0-fs0",
            "Film_cw_v1-seed1-fs0",
            "Film_cw_v2-seed1-fs0",
            "Film_cw_v1-seed2-fs0",
            "Film_cw_v2-seed2-fs0",
        ],
        "Film_fs5": [
            "Film_cw_v1-seed0-fs5",
            "Film_cw_v2-seed0-fs5",
            "Film_cw_v1-seed1-fs5",
            "Film_cw_v2-seed1-fs5",
            "Film_cw_v1-seed2-fs5",
            "Film_cw_v2-seed2-fs5",
        ],
        "concat1_fs12": [
            "baseline-concat1_cw1_v1-seed0-fs12",
            "baseline-concat1_cw1_v2-seed0-fs12",
            "baseline-concat1_cw1_v1-seed1-fs12",
            "baseline-concat1_cw1_v2-seed1-fs12",
            "baseline-concat1_cw1_v1-seed2-fs12",
            "baseline-concat1_cw1_v2-seed2-fs12",
        ],
        "concat1_fs8": [
            "baseline-concat1_cw1_v1-seed0-fs8",
            "baseline-concat1_cw1_v2-seed0-fs8",
            "baseline-concat1_cw1_v1-seed1-fs8",
            "baseline-concat1_cw1_v2-seed1-fs8",
            "baseline-concat1_cw1_v1-seed2-fs8",
            "baseline-concat1_cw1_v2-seed2-fs8",
        ],
        "concat1_fs0": [
            "baseline-concat1_cw1_v1-seed0-fs0",
            "baseline-concat1_cw1_v2-seed0-fs0",
            "baseline-concat1_cw1_v1-seed1-fs0",
            "baseline-concat1_cw1_v2-seed1-fs0",
            "baseline-concat1_cw1_v1-seed2-fs0",
            "baseline-concat1_cw1_v2-seed2-fs0",
        ],
        "concat1_fs5": [
            "baseline-concat1_cw1_v1-seed0-fs5",
            "baseline-concat1_cw1_v2-seed0-fs5",
            "baseline-concat1_cw1_v1-seed1-fs5",
            "baseline-concat1_cw1_v2-seed1-fs5",
            "baseline-concat1_cw1_v1-seed2-fs5",
            "baseline-concat1_cw1_v2-seed2-fs5",
        ],
        "tabular_fs12": [
            "baseline-tabular_embd8_v1-seed0-fs12",
            "baseline-tabular_embd8_v2-seed0-fs12",
            "baseline-tabular_embd8_v1-seed1-fs12",
            "baseline-tabular_embd8_v2-seed1-fs12",
            "baseline-tabular_embd8_v1-seed2-fs12",
            "baseline-tabular_embd8_v2-seed2-fs12",
        ],
        "tabular_fs8": [
            "baseline-tabular_embd8_v1-seed0-fs8",
            "baseline-tabular_embd8_v2-seed0-fs8",
            "baseline-tabular_embd8_v1-seed1-fs8",
            "baseline-tabular_embd8_v2-seed1-fs8",
            "baseline-tabular_embd8_v1-seed2-fs8",
            "baseline-tabular_embd8_v2-seed2-fs8",
        ],
        "tabular_fs0": [
            "baseline-tabular_embd8_v1-seed0-fs0",
            "baseline-tabular_embd8_v2-seed0-fs0",
            "baseline-tabular_embd8_v1-seed1-fs0",
            "baseline-tabular_embd8_v2-seed1-fs0",
            "baseline-tabular_embd8_v1-seed2-fs0",
            "baseline-tabular_embd8_v2-seed2-fs0",
        ],
        "tabular_fs5": [
            "baseline-tabular_embd8_v1-seed0-fs5",
            "baseline-tabular_embd8_v2-seed0-fs5",
            "baseline-tabular_embd8_v1-seed1-fs5",
            "baseline-tabular_embd8_v2-seed1-fs5",
            "baseline-tabular_embd8_v1-seed2-fs5",
            # "baseline-tabular_embd8_v2-seed2-fs5",
        ],

        "resnet": [
            "baseline-resnet_cw1_v1-seed0-fs12",
            "baseline-resnet_cw1_v2-seed0-fs12",
            "baseline-resnet_cw1_v1-seed1-fs12",
            "baseline-resnet_cw1_v2-seed1-fs12",
            "baseline-resnet_cw1_v1-seed2-fs12",
            "baseline-resnet_cw1_v2-seed2-fs12",
        ],
    }

    # eval_type = "val"  # test or val
    eval_type = "test"  # test or val
    project_name += "_test"

    experiments_names = [

    ]

    # --------------  features set 15 -----------------------
    # fs = "15"
    #
    # versions_seeds = np.array(list(itertools.product([0, 1, 2], [1, 2, 3, 4, 5, 6])) + list(itertools.product([3, 4], [1, 2, 3])))
    # hyper_idxs = [23,  7,  5,  4,  3,  1, 12, 13, 14, 15,  2, 17, 18, 19, 20, 21, 22, 0]
    # daft_idxs = [17, 11,  2, 22, 21,  5,  6,  7,  8,  9, 10, 20, 12, 13, 19, 18, 16, 23]
    # hyper11d14 = [f"TabularAsHyper_embd_trainedTabular8_cw11d14_v{v}-seed{s}-fs15" for s, v in versions_seeds[hyper_idxs]]
    # daft11d14 = [f"DAFT_cw11d14_v{v}-seed{s}-fs15" for s, v in versions_seeds[daft_idxs]]
    # # hyper11d14 = [f"TabularAsHyper_embd_trainedTabular8_cw11d14_v{v}-seed{s}-fs15" for s, v in versions_seeds]
    # # daft11d14 = [f"DAFT_cw11d14_v{v}-seed{s}-fs15" for s, v in versions_seeds]
    #
    # versions_seeds = list(itertools.product([0, 1, 2], [1, 2, 3]))
    # film11d14 = [f"Film_cw11d14_v{v}-seed{s}-fs15" for s, v in versions_seeds]
    # concat11_08_14 = [f"baseline-concat1_cw11_08_14_v{v}-seed{s}-fs15" for s, v in versions_seeds]
    # baseline_image = [f"baseline-resnet_cw1_v{v}-seed{s}-fs15" for s, v in versions_seeds]
    # baseline_tabular = [f"baseline-tabular_embd8_v{v}-seed{s}-fs15" for s, v in versions_seeds]
    #
    # experiments_names += hyper11d14 + [[]]
    # experiments_names += daft11d14 + [[]]
    # experiments_names += film11d14 + [[]]
    # experiments_names += concat11_08_14 + [[]]
    # experiments_names += baseline_image + [[]]
    # experiments_names += baseline_tabular + [[]]


    # experiments_names_dict = {
    #     "Hypernetwork": hyper11d14,
    #     "DAFT": daft11d14,
    #     "FiLM": film11d14,
    #     "Concatenation": concat11_08_14,
    #     "Image only": baseline_image,
    #     "Tabular only": baseline_tabular,
    # }

    # --------------  features set 16/17/18 -----------------------
    fs = "16"
    versions_seeds = list(itertools.product([0, 1, 2], [1, 2, 3]))

    hyper11d14 = [f"TabularAsHyper_embd_trainedTabular8_cw11d14_v{v}-seed{s}-fs{fs}" for s, v in versions_seeds]
    daft11d14 = [f"DAFT_cw11d14_v{v}-seed{s}-fs{fs}" for s, v in versions_seeds]
    film11d14 = [f"Film_cw11d14_v{v}-seed{s}-fs{fs}" for s, v in versions_seeds]
    concat11_08_14 = [f"baseline-concat1_cw11_08_14_v{v}-seed{s}-fs{fs}" for s, v in versions_seeds]
    baseline_image = [f"baseline-resnet_cw1_v{v}-seed{s}-fs15" for s, v in list(itertools.product([0, 1, 2], [1, 2, 3]))]
    baseline_tabular = [f"baseline-tabular_embd8_v{v}-seed{s}-fs{fs}" for s, v in versions_seeds]

    experiments_names += hyper11d14 + [[]]
    experiments_names += daft11d14 + [[]]
    experiments_names += film11d14 + [[]]
    experiments_names += concat11_08_14 + [[]]
    experiments_names += baseline_image + [[]]
    experiments_names += baseline_tabular + [[]]


    # ------------------- location ablation - feature set 12 ----------------
    # fs = "12"
    # experiments_names = []
    # hyper11d14 = [f"TabularAsHyper_embd_trainedTabular8_cw11d14_v{v}-seed{s}-fs{fs}" for s, v in versions_seeds]
    # daft11d14 = [f"DAFT_cw11d14_v{v}-seed{s}-fs{fs}" for s, v in versions_seeds]
    # film11d14 = [f"Film_cw11d14_v{v}-seed{s}-fs{fs}" for s, v in versions_seeds]
    # concat11_08_14 = [f"baseline-concat1_cw11_08_14_v{v}-seed{s}-fs{fs}" for s, v in versions_seeds]
    # baseline_image = [f"baseline-resnet_cw1_v{v}-seed{s}-fs15" for s, v in list(itertools.product([0, 1, 2], [1, 2, 3]))]
    # baseline_tabular = [f"baseline-tabular_embd8_v{v}-seed{s}-fs{fs}" for s, v in versions_seeds]
    #
    # R_R_FFT_FFT_FF = [
    #     "T_AsHyper_R_R_FFT_FFT_FF_embd_trainedTabular8_cw085_v2-seed0-fs12",
    #     "T_AsHyper_R_R_FFT_FFT_FF_cw075085_embd_trainedTabular8_v2-seed0-fs12"
    #     ]
    #
    # experiments_names += hyper11d14 + [[]]
    # experiments_names += daft11d14 + [[]]
    # experiments_names += film11d14 + [[]]
    # experiments_names += concat11_08_14 + [[]]
    # experiments_names += baseline_image + [[]]
    # experiments_names += baseline_tabular + [[]]


    # ----------------------------------------------------------------------------------------------
    # ------------------------------- all metrics inclusive table ----------------------------------
    # ----------------------------------------------------------------------------------------------
    if True:
        all_metrics = pd.DataFrame(columns=["experiment_name"] + metrics_names)
        for experiment_name in experiments_names:
            d = get_experiment_metrics(
                project_name=project_name,

                # regex syntax:   https://dev.mysql.com/doc/refman/8.0/en/regexp.html
                experiment_name=experiment_name,
                regex=True,
                eval_type=eval_type,  # test or val
                metrics_names=metrics_names  # optional metrics: balanced_acc, precision, AUC, f1_macro, f1_micro
            )
            print(d)
            all_metrics = pd.concat([all_metrics, pd.DataFrame([d])], ignore_index=True)

        # print(all_metrics)
        # all_metrics.to_csv(f"/media/oldrrtammyfs/Users/daniel/hyper_table_{eval_type}.csv", index=False)

        all_metrics = arange_results_csv(all_metrics)
        print(all_metrics[["balanced_acc", "precision", "CN_acc", "MCI_acc", "AD_acc"]])
        all_metrics.to_csv(f"/media/rrtammyfs/Users/daniel/full_results_{eval_type}_fs{fs}.csv")

    # ----------------------------------------------------------------------------------------------
    # ----------------------- statistical significance, mean and std values -------------------------
    # ----------------------------------------------------------------------------------------------
    # a = experiments_names_dict["DAFT_fs12"]
    # b = experiments_names_dict["hyper_fs12"]
    #
    # check_statistics(
    #     project_name=project_name,
    #     experiment_name_a=a,
    #     experiment_name_b=b,
    #     eval_type=eval_type,  # test or val
    #     metric_name="balanced_acc",  # optional metrics: balanced_acc, precision, AUC, f1_macro, f1_micro
    # )
    # check_statistics(
    #     project_name=project_name,
    #     experiment_name_a=a,
    #     experiment_name_b=b,
    #     eval_type=eval_type,  # test or val
    #     metric_name="precision",  # optional metrics: balanced_acc, precision, AUC, f1_macro, f1_micro
    # )
    # check_statistics(
    #     project_name=project_name,
    #     experiment_name_a=a,
    #     experiment_name_b=b,
    #     eval_type=eval_type,  # test or val
    #     metric_name="AUC",  # optional metrics: balanced_acc, precision, AUC, f1_macro, f1_micro
    # )
    # check_statistics(
    #     project_name=project_name,
    #     experiment_name_a=a,
    #     experiment_name_b=b,
    #     eval_type=eval_type,  # test or val
    #     metric_name="f1_macro",  # optional metrics: balanced_acc, precision, AUC, f1_macro, f1_micro
    # )
    if False:
        # experiments = ["tabular", "concat1", "Film", "DAFT", "hyper"]
        # features_sets = [12, 8, 5]
        experiments = ["DAFT", "hyper"]
        features_sets = [12]

        significance_dict = {"experiment": [], "features_set": []}
        for metric_name in metrics_names:
            significance_dict[metric_name] = []
            # significance_dict[f"p_val-{metric_name}"] = []
        for features_set in features_sets:
            for exp_name in experiments:
                significance_dict["experiment"].append(exp_name)
                significance_dict["features_set"].append(features_set)
                base2compare = f"hyper_fs{features_set}"
                exp_name += f"_fs{features_set}"
                for metric_name in metrics_names:
                    base_meanNstd, exp_meanNstd, p_value = check_statistics(
                        project_name=project_name,

                        # regex syntax:   https://dev.mysql.com/doc/refman/8.0/en/regexp.html
                        experiment_name_a=experiments_names_dict[base2compare],
                        experiment_name_b=experiments_names_dict[exp_name],
                        eval_type=eval_type,  # test or val
                        metric_name=metric_name,  # optional metrics: balanced_acc, precision, AUC, f1_macro, f1_micro
                    )

                    if not ("hyper" in exp_name):
                        significance_dict[metric_name].append(exp_meanNstd + f" (p={p_value.round(3)})")
                    else:
                        significance_dict[metric_name].append(exp_meanNstd)

        statistical_df = pd.DataFrame(significance_dict)
        statistical_df.to_csv(f"/media/rrtammyfs/Users/daniel/statistical_table_{eval_type}.csv", index=False)


    # ----------------------------------------------------------------------------------------------
    # ------------------------------------- confusion matrix ---------------------------------------
    # ----------------------------------------------------------------------------------------------
    if False:
        for experiment_name in experiments_names_dict.keys():
            save_path = f"/media/rrtammyfs/Users/daniel/research results/confusion_mat/{experiment_name}_{eval_type}.png"
            conf_mat = get_experiment_confmat(
                project_name=project_name,

                # regex syntax:   https://dev.mysql.com/doc/refman/8.0/en/regexp.html
                experiment_name=experiments_names_dict[experiment_name],
                regex=True,
                eval_type=eval_type,  # test or val
                save_path=save_path
            )
