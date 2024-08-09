from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
