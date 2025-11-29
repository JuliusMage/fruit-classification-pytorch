# src/evaluate.py
"""
Evaluation script: loads best model from output directory and runs evaluation
on validation/test split defined in dataset.get_dataloaders.

Usage example:
  python src/evaluate.py --data-dir ../data --model-path ../results/training_logs/best_model.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_dataloaders, set_seed
from model_cnn import SimpleCNN
from model_mlp import MLPClassifier


def load_model_from_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    args = ckpt.get("args", None)
    class_map = ckpt.get("class_map", None)

    # Try to infer which model type was used from args
    model_type = None
    if args:
        model_type = args.get("model", None)

    # Fall back to CNN if unknown
    if model_type == "mlp":
        model = MLPClassifier(input_channels=3, img_size=args.get("img_size", 300),
                              hidden=args.get("hidden", 512), num_classes=len(class_map) if class_map else 3)
    else:
        model = SimpleCNN(in_channels=3, num_classes=len(class_map) if class_map else 3)

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, class_map


def evaluate_model(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)
    return running_loss / total, correct / total


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    train_loader, val_loader, class_map = get_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers
    )

    model, ckpt_class_map = load_model_from_checkpoint(args.model_path, device)
    if ckpt_class_map:
        print("Model class map from checkpoint:", ckpt_class_map)

    print("Evaluating on validation set...")
    val_loss, val_acc = evaluate_model(model, val_loader, device)
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Optionally evaluate on train for sanity check
    if args.eval_train:
        print("Evaluating on training set...")
        train_loss, train_acc = evaluate_model(model, train_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=309)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-train", action="store_true", help="Also evaluate on training set for sanity check.")
    args = parser.parse_args()

    main(args)
