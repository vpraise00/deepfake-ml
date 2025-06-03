# src/scripts/train.py

import os
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import KFold

from src.datasets.ffpp_dataset import FFPPFrameDataset
from src.models.deepfake_adapter import DeepfakeAdapter
from src.utils.metrics import compute_auc, compute_eer
from src.utils.transforms import get_ffpp_transforms


def set_seed(seed: int = 42):
    """
    재현성을 위해 모든 랜덤 시드 고정.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config_path: str):
    # 1) Config 로드
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) 전체 Train+Val 데이터셋 로드 (split="trainval")
    full_dataset = FFPPFrameDataset(config_path, split="trainval")
    num_samples = len(full_dataset)
    print(f"Total Train+Val samples: {num_samples}")

    # 3) K-Fold 세팅 (예: 5-Fold)
    k_folds = cfg["training"].get("k_folds", 5)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # 4) K-Fold loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(num_samples))):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        # (A) Subset 생성
        train_subset = Subset(full_dataset, train_idx)
        val_subset   = Subset(full_dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=True
        )

        # (B) 모델, 손실함수, 옵티, 스케줄러 초기화
        model = DeepfakeAdapter(cfg).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg["optimizer"]["lr"],
            weight_decay=cfg["optimizer"]["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["scheduler"]["T_max"]
        )

        # Fold별 디렉터리 생성
        fold_ckpt_dir = os.path.join(cfg["output_dir"], f"fold_{fold+1}")
        fold_log_dir  = os.path.join(cfg["log_dir"], f"fold_{fold+1}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)
        os.makedirs(fold_log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=fold_log_dir)
        best_auc = 0.0

        # (C) Epoch loop
        for epoch in range(cfg["training"]["epochs"]):
            #########################
            # Training 단계
            #########################
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            scheduler.step()
            avg_train_loss = running_loss / len(train_loader.dataset)
            writer.add_scalar("Loss/train", avg_train_loss, epoch)

            #########################
            # Validation 단계
            #########################
            model.eval()
            all_labels = []
            all_scores = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)[:, 1]  # fake 확률
                    all_labels.extend(labels.cpu().numpy())
                    all_scores.extend(probs.cpu().numpy())

            val_auc = compute_auc(np.array(all_labels), np.array(all_scores))
            writer.add_scalar("AUC/val", val_auc, epoch)

            print(
                f"[Fold {fold+1}/{k_folds} | Epoch {epoch+1}/{cfg['training']['epochs']}] "
                f"Train Loss: {avg_train_loss:.4f}  Val AUC: {val_auc:.4f}"
            )

            #########################
            # 최적 모델 저장
            #########################
            if val_auc > best_auc:
                best_auc = val_auc
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auc": val_auc,
                }
                save_path = os.path.join(fold_ckpt_dir, "best.pth")
                torch.save(ckpt, save_path)
                print(f"→ New best model for Fold {fold+1} saved at {save_path}")

        writer.close()

    # 5) K-Fold가 모두 끝난 뒤, Test 셋 평가 (가장 마지막 Fold 모델 사용)
    print("\n=== K-Fold Training Complete ===")
    print("Evaluating on the Test set using the last Fold's best model...")

    # (A) Test Dataset/Loader 생성
    test_dataset = FFPPFrameDataset(config_path, split="test")
    test_loader  = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )
    print(f"Test samples: {len(test_dataset)}")

    # (B) 마지막 Fold 모델 로드
    last_fold_dir = os.path.join(cfg["output_dir"], f"fold_{k_folds}")
    last_ckpt_path = os.path.join(last_fold_dir, "best.pth")
    assert os.path.exists(last_ckpt_path), f"Checkpoint not found: {last_ckpt_path}"

    model = DeepfakeAdapter(cfg).to(device)
    ckpt = torch.load(last_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # (C) Test 셋 예측
    all_test_labels = []
    all_test_scores = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]  # fake 확률

            all_test_labels.extend(labels.cpu().numpy())
            all_test_scores.extend(probs.cpu().numpy())

    test_auc = compute_auc(np.array(all_test_labels), np.array(all_test_scores))
    test_eer = compute_eer(np.array(all_test_labels), np.array(all_test_scores))
    print(f"Final Test AUC: {test_auc:.4f}")
    print(f"Final Test EER: {test_eer:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train+CV DeepfakeAdapter on FFPP")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ffpp_c23.yaml",
        help="Path to the config file"
    )
    args = parser.parse_args()

    main(args.config)
