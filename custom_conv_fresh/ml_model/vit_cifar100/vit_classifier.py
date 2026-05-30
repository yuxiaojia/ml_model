"""
Run ViT-CIFAR100 inference on the entire CIFAR-100 test set (10,000 images).
Model: https://huggingface.co/Ahmed9275/Vit-Cifar100

Usage:
    python classifier.py
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np

MODEL_NAME = "Ahmed9275/Vit-Cifar100"
BATCH_SIZE = 100
DATA_DIR   = "./data"


def get_dataloader():
    # CIFAR-100 images are 32x32; ViT expects 224x224
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True), dataset


def tensor_to_pil_batch(batch: torch.Tensor) -> list:
    """Convert a batch of (C,H,W) float tensors to a list of PIL Images."""
    images = []
    for t in batch:
        arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print("Loading CIFAR-100 test set...")
    loader, dataset = get_dataloader()

    correct = 0
    total   = 0

    print(f"Running inference on {len(dataset)} images (batch size {BATCH_SIZE})...\n")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            pil_images = tensor_to_pil_batch(images)
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            logits = model(**inputs).logits        # (B, 100)
            preds  = logits.argmax(dim=-1).cpu()   # predicted class ids

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            if (batch_idx + 1) % 10 == 0 or total == len(dataset):
                acc = correct / total * 100
                print(f"  [{total:>5}/{len(dataset)}]  Running accuracy: {acc:.2f}%")

    final_acc = correct / total * 100
    print(f"\nFinal accuracy on CIFAR-100 test set: {final_acc:.2f}%  ({correct}/{total})")


if __name__ == "__main__":
    run()
