"""ResNet-50 / ViT-B/16 classifiers: khởi tạo scratch hoặc pretrained, cơ chế freeze, nhóm tham số cho AdamW."""
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

from .config import TRAINABLE_MODULES

# Ghi rõ phiên bản trọng số, không dùng alias "DEFAULT" (alias có thể đổi giữa các bản TorchVision)
PRETRAINED_WEIGHTS = {
    "resnet50": ResNet50_Weights.IMAGENET1K_V1,   # top-1 ImageNet 76.1%
    "vit_b_16": ViT_B_16_Weights.IMAGENET1K_V1,   # top-1 ImageNet 81.1%
}
HEAD_PREFIX = {"resnet50": "fc.", "vit_b_16": "heads."}


def build_model(arch: str, num_classes: int, init: str = "scratch",
                head_std: float = 0.01, head_seed: int = 0) -> nn.Module:
    """Kiến trúc TorchVision (weights=None hoặc IMAGENET1K_V1) + head Linear mới cho `num_classes` lớp.

    Head được khởi tạo bằng RNG riêng (fork_rng) -> giống hệt nhau giữa scratch và pretrained cùng seed.
    """
    if init not in ("scratch", "pretrained"):
        raise ValueError(f"Unsupported init: {init}")
    weights = PRETRAINED_WEIGHTS[arch] if init == "pretrained" else None

    if arch == "resnet50":
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)                   # 2048 -> 101
        head = model.fc
    elif arch == "vit_b_16":
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)   # 768 -> 101
        head = model.heads.head
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(head_seed)
        nn.init.normal_(head.weight, std=head_std)
        nn.init.zeros_(head.bias)
    return model


def apply_freeze(model: nn.Module, arch: str, freeze: str) -> list:
    """Đặt requires_grad theo chế độ freeze; trả về (và lưu vào model.frozen_modules) các module con bị freeze."""
    modes = TRAINABLE_MODULES[arch]
    if freeze not in modes:
        raise ValueError(f"Freeze mode '{freeze}' is not defined for {arch}")
    keep = modes[freeze]

    def is_trainable(name: str) -> bool:
        return keep is None or any(name == k or name.startswith(k + ".") for k in keep)

    for name, p in model.named_parameters():
        p.requires_grad = is_trainable(name)

    frozen = [name for name, _ in model.named_children() if not is_trainable(name)]
    model.frozen_modules = frozen
    return frozen


def set_train_mode(model: nn.Module) -> None:
    """model.train() cho phần trainable, nhưng giữ module bị freeze ở eval() (BatchNorm không cập nhật running stats)."""
    model.train()
    for name in getattr(model, "frozen_modules", []):
        model.get_submodule(name).eval()


def param_groups(model: nn.Module, arch: str, lr_head: float, backbone_mult: float, weight_decay: float) -> list:
    """4 nhóm {backbone, head} x {decay, no_decay}; bỏ tham số bị freeze và nhóm rỗng.

    no_decay: bias, tham số norm (ndim <= 1), pos_embedding và class_token của ViT.
    """
    groups = {(part, decay): [] for part in ("backbone", "head") for decay in (True, False)}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if name.startswith(HEAD_PREFIX[arch]) else "backbone"
        no_decay = p.ndim <= 1 or "pos_embedding" in name or "class_token" in name
        groups[(part, not no_decay)].append(p)

    lr = {"head": lr_head, "backbone": lr_head * backbone_mult}
    return [
        {"name": f"{part}_{'decay' if decay else 'no_decay'}", "params": ps,
         "lr": lr[part], "weight_decay": weight_decay if decay else 0.0}
        for (part, decay), ps in groups.items() if ps
    ]


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
