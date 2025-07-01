import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchinfo import summary
from PIL import Image
import os
from tqdm import tqdm
from typing import Dict, List, Tuple
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
import warnings
warnings.filterwarnings('ignore')
# Hàm train_step với mixed precision
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int,
               num_epochs: int) -> Tuple[float, float]:
    scaler = GradScaler()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000")
    model.train()
    train_loss, train_acc = 0, 0
    total, correct = 0, 0
    for batch, (images, labels) in enumerate(pbar):
        if images is None or labels is None:
            continue
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            label_pred = model(images)
            loss = criterion(label_pred, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        _, predicted = torch.max(label_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/(batch+1):.4f}")
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(dataloader):.4f}')
    train_loss /= len(dataloader)
    train_acc = 100 * correct / total if total > 0 else 0
    return train_loss, train_acc

# Hàm test_step với mixed precision
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0
    correct = 0
    total = 0
    with torch.no_grad():
        with autocast():
            for images, labels in dataloader:
                if images is None or labels is None:
                    continue
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    test_loss /= len(dataloader)
    test_acc = 100 * correct / total if total > 0 else 0
    return test_loss, test_acc

# Hàm train chính với scheduler và early stopping
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_test_loss = float('inf')
    patience = 5
    patience_counter = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, criterion, optimizer, device, epoch, epochs)
        test_loss, test_acc = test_step(model, test_dataloader, criterion, device)
        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        scheduler.step(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'clip_finetuned.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return results