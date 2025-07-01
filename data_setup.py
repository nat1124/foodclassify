import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

def pred_df(path: str, class_dic: dict) -> pd.DataFrame:
    array = open(path, "r").read().splitlines()
    image_path = "Data/food-101/food-101/images/"
    full_path = [image_path + img + ".jpg" for img in array]
    labels = [img.split('/')[0] for img in array]
    imgs = pd.DataFrame({
        'path': full_path,
        'label': labels,
        'label_idx': [class_dic[label] for label in labels]
    })
    imgs = shuffle(imgs, random_state=42)
    return imgs

class CutOut(object):
    def __init__(self, mask_size, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() > self.p:
            return img
        h, w = img.size(-2), img.size(-1)
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()
        y1 = max(0, y - self.mask_size // 2)
        y2 = min(h, y + self.mask_size // 2)
        x1 = max(0, x - self.mask_size // 2)
        x2 = min(w, x + self.mask_size // 2)
        img[:, y1:y2, x1:x2] = 0
        return img

train_transform = transforms.Compose([
    transforms.RandomRotation(45),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.7),
    #transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    CutOut(mask_size=20, p=0.5)
])

test_transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FoodDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_name = self.dataframe.path.iloc[idx]
        label = self.dataframe['label_idx'].iloc[idx]
        try:
            image = Image.open(img_name)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None, None