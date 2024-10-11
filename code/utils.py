from torch import Tensor
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional
import matplotlib.pyplot as plt
import numpy as np


def readImage(path: 'str', size: 'tuple' = (224, 224)) -> 'Tensor':
    '''Return a Tensor in shape C x H x W, normalized.'''
    img = Image.open(path)

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.PILToTensor()
    ])

    return transform(img)/255


def show(imgs: 'Tensor | list[Tensor]'):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
