# kindset-package-vxl

kindset-package-vxl 是一个用于处理 EMNIST、CIFAR10 数据集的实用工具。该工具提供了两个主要功能：

1. **提取特定类别子集：** `extract_kinds_dataset` 函数提取数据集的子集，仅包含用户指定的类别。
2. **可视化数据集：** `visual_paint_img` 函数用于可视化数据集中的图像。

## Installation

You can install the package using pip:

```bash
pip install -i https://test.pypi.org/simple/ kindset-package-vxl==0.0.4
```

To use this utility, you need to have the necessary dependencies installed. You can install torch [ and matplotlib if you want to see the painting of images ].

## Usage

```python
'''Import the necessary functions'''
from kindset_package_vxl import kindset

'''Load the dataset'''
train_set = EMNIST(root='./digits', train=True, split='digits', transform=transforms.ToTensor())

'''Extract a subset based on your criteria (e.g., class label)'''
# kinds = [1, 3, 5, 7] list
# kinds = (1, 5, 7) tuple
# kinds = 9 int
kind_set = kindset.extract_kinds_dataset(train_set, kinds=9)

'''Visualize a batch of images from the subset'''
kindset.visual_paint_img(kind_set,batch_size=10,shuffle=True)
```

## Example

```py
from kindset_package_vxl import kindset
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms

train_set = EMNIST(root='./digits', train=True, split='digits', transform=transforms.ToTensor())
# train_set = CIFAR10(root='./cifar', train=True, transform=transforms.ToTensor())

kind_set = kindset.extract_kinds_dataset(train_set, kinds = (1, 5, 7))
print("Subset indices:", kind_set.indices)
kindset.visual_paint_img(kind_set,batch_size=10,shuffle=True)
```

![image](https://github.com/vxlot/getdata/assets/151598803/5e6cc64d-6931-412c-80d7-e39fa2f6ff92)


## Author

Email: weixiaolu617@gmail.com
