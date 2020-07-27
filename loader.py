import numpy as np
import torch
import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

from PIL import Image

IMG_EXTENSIONS = [
    '.png',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir):
    images = []
    for filename in sorted(os.listdir(dir)):
        if filename.split('.')[-1] != "jpg":
            continue
        # filename : [age]_[gender]_[race]_[date&time].jpg
        path = os.path.join(dir, filename)
        target = float(filename.split('_')[0])
        images.append((path, target))
    return images


def transform_image(img, random):
    resize = transforms.Resize((64, 64))
    crop = transforms.RandomCrop(64, padding=4)
    flip = transforms.RandomHorizontalFlip()
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([resize, crop, flip, totensor, normalize])
    if not random:
        transform = transforms.Compose([resize, totensor, normalize])
    return transform(img)


class MNIST(data.Dataset):

    def __init__(self, root, random = True):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.random = random

    def __getitem__(self, idx : int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input_tensor, depth_tensor)
        """
        path, target = self.imgs[idx]
        target = torch.tensor([target], requires_grad = False)
        input_tensor = transform_image(Image.open(path).convert("RGB"), random = self.random)
        return input_tensor, target

    def __len__(self):
        return len(self.imgs)
