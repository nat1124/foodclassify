import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from pathlib import Path
import clip
from torch import optim
import json
import warnings
warnings.filterwarnings('ignore')

from data_setup import pred_df, CutOut, train_transform, test_transform, FoodDataset
from engine import train_step, test_step, train
from model_builder import CLIPFineTuner



if __name__ == '__main__':
    data_path = Path("Data/food-101/")
    image_path = data_path / "food-101"

    class_names = open(f"{image_path / 'meta'}\\classes.txt", "r").read().splitlines()
    train_images = open(f"{image_path / 'meta'}\\train_reduced.txt", "r").read().splitlines()
    test_images = open(f"{image_path / 'meta'}\\test_reduced.txt", "r").read().splitlines()

    class_dic = {name: idx for idx, name in enumerate(class_names)}

    train_dir = image_path / "meta/train_reduced.txt"
    test_dir = image_path / "meta/test_reduced.txt"
    train_imgs = pred_df(train_dir, class_dic)
    test_imgs = pred_df(test_dir, class_dic)
    # Cấu hình DataLoader
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    train_dataset = FoodDataset(train_imgs, transform=train_transform)
    test_dataset = FoodDataset(test_imgs, transform=test_transform)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        pin_memory=True
    )

    # Tải mô hình CLIP
    model, preprocess = clip.load("ViT-B/32", jit=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze tham số visual của CLIP
    for param in model.visual.parameters():
        param.requires_grad = False

    # Định nghĩa mô hình fine-tune
    num_classes = len(class_names)
    model_ft = CLIPFineTuner(model, num_classes).to(device)
    
    # Cấu hình optimizer và loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)
    
    # Train mô hình
    num_epochs = 60
    model_ft_results = train(
        model=model_ft,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=num_epochs,
        device=device
    )

    # Lưu kết quả train
    with open('train_results.json', 'w') as f:
        json.dump(model_ft_results, f)