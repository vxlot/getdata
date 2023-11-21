from matplotlib import pyplot as plt
import torch
from torch.utils.data import Subset


def mark_kinds_pos_in_labels(kinds, labels):
    """
    Marks positions in labels that match the specified kinds.

    Parameters:
    - kinds (int, tuple, or list): The class labels to mark.
    - labels (torch.Tensor): The input labels.

    Returns:
    torch.Tensor: A boolean tensor indicating positions where labels match the specified kinds.
    """
    if isinstance(kinds, int):
        return torch.tensor(labels == kinds).squeeze()
    if isinstance(kinds, tuple) or isinstance(kinds, list):
        # 将labels和kinds张量转换为1维张量
        kinds = torch.tensor(kinds)
        return torch.tensor(labels.view(-1, 1).eq(kinds).any(dim=1)).squeeze()
    return None

def get_all_labels(dataset):
    """
    Gets all labels from the dataset.

    Parameters:
    - dataset: The input dataset.

    Returns:
    torch.Tensor: A tensor containing all labels in the dataset.
    """
    name = dataset.__class__.__name__.lower()

    if 'mnist' in name or 'cifar' in name:
        return torch.tensor(dataset.targets)
    else:
        raise KeyError
    
def extract_kinds_dataset(dataset, kinds, labels=None):
    """
    Extracts a subset from the dataset based on specified kinds.

    Parameters:
    - dataset: The input dataset.
    - kinds (int, tuple, or list): The class labels to extract.
    - labels (torch.Tensor, optional): The input labels. If not provided, all labels from the dataset will be used.

    Returns:
    torch.utils.data.Subset: A subset of the input dataset containing only samples with specified kinds.
    """
    if labels is None:
        labels = get_all_labels(dataset)

    mask = mark_kinds_pos_in_labels(kinds, labels)
    indices = torch.nonzero(mask).squeeze()
    dataset = Subset(dataset, indices)
    return dataset

def visual_paint_img(data_set,batch_size,shuffle):
    """
    Displays a batch of images from a data loader.

    Parameters:
    - data_loader: The input data loader.
    - batch_size (int): The size of the image batch to display.
    - shuffle: Change the order.
    """
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    images, labels = next(iter(data_loader))
    images = (images.numpy() * 0.5) + 0.5
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 3))
    for i in range(batch_size):
        axes[i].imshow(images[i][0], cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    plt.show()


