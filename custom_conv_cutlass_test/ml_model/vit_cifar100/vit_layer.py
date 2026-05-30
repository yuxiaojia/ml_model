"""
Print all layers of the ViT-CIFAR100 model with their shapes and parameter counts.
Model: https://huggingface.co/Ahmed9275/Vit-Cifar100

Usage:
    python vit_layer.py
"""

from transformers import ViTForImageClassification

MODEL_NAME = "Ahmed9275/Vit-Cifar100"


def print_layers(model):
    print(f"\n{'Layer':<70} {'Shape':<30} {'Params':>10}")
    print("-" * 115)

    total_params = 0
    for name, param in model.named_parameters():
        shape = str(list(param.shape))
        n = param.numel()
        total_params += n
        print(f"{name:<70} {shape:<30} {n:>10,}")

    print("-" * 115)
    print(f"{'TOTAL PARAMETERS':<70} {'':30} {total_params:>10,}")


def print_modules(model):
    print("\n=== Module Tree ===")
    print(model)


if __name__ == "__main__":
    print("Loading model...")
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print_modules(model)
    print_layers(model)
