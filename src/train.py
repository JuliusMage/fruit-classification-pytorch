# src/train.py
"""
Train script for MLP or CNN model.

Usage examples:
  python src/train.py --data-dir ../data --model cnn --epochs 10 --batch-size 64 --lr 1e-4
  python src/train.py --data-dir ../data --model mlp --epochs 5 --batch-size 32 --lr 1e-2
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders, set_seed
from model_cnn import SimpleCNN
from model_mlp import MLPClassifier


def save_checkpoint(state, out_dir, name="checkpoint.pth"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / name)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Val/Test", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    train_loader, val_loader, class_map = get_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers
    )
    num_classes = len(class_map)
    print("Found classes:", class_map)

    if args.model.lower() == "cnn":
        model = SimpleCNN(in_channels=3, num_classes=num_classes)
    elif args.model.lower() == "mlp":
        model = MLPClassifier(input_channels=3, img_size=args.img_size, hidden=args.hidden, num_classes=num_classes)
    else:
        raise ValueError("Unknown model: choose 'cnn' or 'mlp'")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unknown optimizer: choose 'sgd' or 'adam'")

    best_val_acc = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - start

        print(f"Epoch {epoch}/{args.epochs} | time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save checkpoint
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "class_map": class_map,
            "args": vars(args)
        }
        save_checkpoint(state, out_dir, name=f"checkpoint_epoch{epoch}.pth")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(state, out_dir, name="best_model.pth")

    print("Training complete. Best val acc:", best_val_acc)
    # Save final
    save_checkpoint({
        "model_state": model.state_dict(),
        "class_map": class_map,
        "args": vars(args)
    }, out_dir, name="final_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory (contains class subfolders).")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "mlp"])
    parser.add_argument("--img-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--hidden", type=int, default=512, help="Hidden size for MLP.")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=309)
    parser.add_argument("--output-dir", type=str, default="../results/training_logs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    main(args)
