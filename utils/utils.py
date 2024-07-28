from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import matplotlib.pyplot as plt
import threading
import matplotlib.pyplot as plt
import seaborn as sns
import torchmetrics
import pandas as pd
from easydict import EasyDict

def get_class_weight(train_loader, valid_loader):
    all_labels = list(train_loader.dataset.metadata.Group) + list(valid_loader.dataset.metadata.Group)
    all_labels = [train_loader.dataset.labels_dict[label] for label in all_labels]
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=list(all_labels))
    return torch.Tensor(class_weights)

def nonsquared_conf_mat(preds, targets ,labels, normalize=None, classes_3=False):
    m = len(targets.unique())
    n = len(preds[0])
    confusion_matrix = np.zeros((m, n))
    preds = preds.argmax(axis=1)
    for p, t in zip(preds, targets):
        confusion_matrix[t, p] += 1

    if classes_3:
        if m < n:
            confusion_matrix[:, 1] += confusion_matrix[:, 3] + confusion_matrix[:, 4]
        # else:
        #     confusion_matrix[1] += confusion_matrix[2] + confusion_matrix[3]
        confusion_matrix = confusion_matrix[:3, :3]

    if confusion_matrix.shape[0] == confusion_matrix.shape[1]:
        precision = (np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)).mean()
        balanced_acc = (np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)).mean()

    if normalize == "true":
        for i in range(confusion_matrix.shape[0]):
            confusion_matrix[i] /= confusion_matrix[i].sum()


    if confusion_matrix.shape[0] == 5:
        confusion_matrix[[0, 1, 2, 3, 4]] = confusion_matrix[[0, 3, 1, 4, 2]]
        labels_y = ["CN", 'EMCI', 'MCI', "LMCI", "AD"]
    elif confusion_matrix.shape[0] == 2:
        labels_y = ["CN", "AD"]
    else:
        labels_y = ["CN", 'MCI', "AD"]

    if confusion_matrix.shape[1] == 5:
        confusion_matrix[:, [0, 1, 2, 3, 4]] = confusion_matrix[:, [0, 3, 1, 4, 2]]
        labels_x = ["CN", 'EMCI', 'MCI', "LMCI", "AD"]
    elif confusion_matrix.shape[1] == 2:
        labels_x = ["CN", "AD"]
    else:
        labels_x = ["CN", 'MCI', "AD"]

    cm = confusion_matrix.copy()
    confusion_matrix = confusion_matrix.round(4)
    plt.figure(figsize=(5, 4))
    fig_ = sns.heatmap(confusion_matrix, annot=True, cmap='Reds', fmt='g').get_figure()
    plt.title('prediction')
    plt.ylabel('Ground Truth')
    plt.xticks(np.arange(confusion_matrix.shape[1]) + 0.5, labels_x)
    plt.yticks(np.arange(confusion_matrix.shape[0]) + 0.5, labels_y)
    fig_.axes[0].xaxis.tick_top()
    if confusion_matrix.shape[0] == confusion_matrix.shape[1]:
        plt.suptitle(f"recall/balanced acc:{balanced_acc:.4f}   precision:{precision:.4f}", y=0.03, va='bottom')

    plt.close(fig_)
    return fig_, cm, precision



def results_df2bars_graph():
    title = 'AD hyperlayer`s position ablation study'
    models = ["b1Do", "b2Do", "b3Do", "b4c1", "2fc", "b4Do(ours)", "w/o hyperlayers"]
    metrics = "BA,PRC,f1,AUC,TP-CN,TP-MCI,TP-AD".split(',')
    results = [
        list(map(lambda x:float(x), '0.638	0.592	0.596	0.801	0.748	0.453	0.714'.split('\t'))),
        list(map(lambda x:float(x), '0.651	0.605	0.613	0.811	0.731	0.488	0.734'.split('\t'))),
        list(map(lambda x:float(x), '0.663	0.619	0.624	0.817	0.751	0.486	0.757'.split('\t'))),
        list(map(lambda x:float(x), '0.638	0.593	0.598	0.803	0.744	0.455	0.714'.split('\t'))),
        list(map(lambda x:float(x), '0.616	0.578	0.587	0.787	0.721	0.468	0.660'.split('\t'))),
        list(map(lambda x:float(x), '0.673	0.624	0.630	0.822	0.759	0.495	0.764'.split('\t'))),
        list(map(lambda x:float(x), '0.603	0.560	0.555	0.775	0.663	0.423	0.726'.split('\t'))),
    ]

    # title = 'AD vs CN'
    # models = ["HyperFusion", "Late-Fusion", "only-tabular", "only-imaging"]
    # metrics = "BA,PRC,f1,AUC,TP-CN,TP-AD".split(',')
    # results = [
    #     list(map(lambda x:float(x), '0.912	0.882	0.892	0.965	0.878	0.945'.split('\t'))),
    #     list(map(lambda x:float(x), '0.897	0.867	0.876	0.958	0.861	0.885'.split('\t'))),
    #     list(map(lambda x:float(x), '0.846	0.825	0.833	0.921	0.847	0.844'.split('\t'))),
    #     list(map(lambda x:float(x), '0.888	0.858	0.868	0.951	0.858	0.917'.split('\t'))),
    # ]

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
        ax.bar(bar_positions[i], results[i], width=barWidth,
               edgecolor='grey', label=feature)

    # Adding Xticks and labels
    # ax.set_xlabel('Categories', fontweight='bold', fontsize=15)
    # ax.set_ylabel('Values', fontweight='bold', fontsize=15)
    ax.set_xticks(xlabel_positions)
    # ax.set_yticks(np.arange(0, 1, 0.05))
    # if ylim is not None:
    #     plt.ylim(ylim[0], ylim[1])
    # plt.ylim(0.4, 1.)
    # plt.ylim(0.7, 1.)
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
    # plt.savefig(f"/media/rrtammyfs/Users/daniel/AD_bars_test_res.png")
    plt.show()

if __name__ == '__main__':
    results_df2bars_graph()