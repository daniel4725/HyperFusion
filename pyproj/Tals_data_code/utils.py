from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_class_weight(train_loader, valid_loader):
    all_labels = list(train_loader.dataset.metadata.DX_bl) + list(valid_loader.dataset.metadata.DX_bl)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=list(all_labels))
    return torch.Tensor(class_weights)

def get_img_conf_matrix(classes, conf_mat):
    y_axis = classes
    x_axis = classes
        
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat)

    # Show all ticks and label them with the respective list entries
    plt.xticks(np.arange(len(x_axis)), labels=x_axis)
    ax.xaxis.tick_top()
    plt.yticks(np.arange(len(y_axis)), labels=y_axis)
    plt.title('prediction')
    plt.ylabel('GT')

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_axis)):
        for j in range(len(x_axis)):
            text = ax.text(j, i, conf_mat[i, j],
                        ha="center", va="center", color="w")

    # ax.set_title("Confusion Matrix")
    fig.tight_layout()

    fig.tight_layout(pad=0)

    # To remove the huge white borders
    ax.margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.show()
    return image_from_plot