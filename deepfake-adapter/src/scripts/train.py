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

from sklearn.model_selection import train_test_split, KFold

from src.datasets.ffpp_dataset import FFPPFrameDataset
from src.models.deepfake_adapter import DeepfakeAdapter
from src.utils.metrics import compute_auc, compute_eer


def set_seed(seed: int = 42):
    """
    재현성을 위해 모든 랜덤 시드 고정.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config_path: str):
    # 1) 설정 로드
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) 전체 데이터셋 로드 (alltype)
    full_dataset = FFPPFrameDataset(config_path)
    N = len(full_dataset)
    print(f"Total samples (alltype): {N}")

    # 3) 전체 → Train(80%) / Test(20%) 분할 (stratify)
    all_indices = list(range(N))
    all_labels = list(full_dataset.labels)

    trainval_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.2,            # 20%를 Test로 분리
        random_state=42,
        stratify=all_labels
    )
    print(f"Train+Val count: {len(trainval_idx)}, Test count: {len(test_idx)}")

    # 4) Train+Val(80%) → Train(70%) / Val(10%) 분할
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.125,  # 0.125 * 0.8 = 0.10 전체
        random_state=42,
        stratify=[all_labels[i] for i in trainval_idx]
    )
    print(f"Train count: {len(train_idx)}, Val count: {len(val_idx)}, Test count: {len(test_idx)}")

    # 5) Train(70%)에 대해 K-Fold 적용 (예: 5-Fold)
    k_folds = cfg["training"].get("k_folds", 5)
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (tr_sub, va_sub) in enumerate(kfold.split(train_idx)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        # 상대 인덱스를 실제 인덱스로 매핑
        true_train_idx = [train_idx[i] for i in tr_sub]
        true_val_idx   = [train_idx[i] for i in va_sub]

        train_subset = Subset(full_dataset, true_train_idx)
        val_subset   = Subset(full_dataset, true_val_idx)

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

        # (A) 모델, 손실함수, 옵티마이저, 스케줄러 정의
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

        # (B) Fold별 체크포인트/로그 디렉터리 생성
        fold_ckpt_dir = os.path.join(cfg["output_dir"], f"fold_{fold+1}")
        fold_log_dir  = os.path.join(cfg["log_dir"],  f"fold_{fold+1}")
        os.makedirs(fold_ckpt_dir, exist_ok=True)
        os.makedirs(fold_log_dir,  exist_ok=True)

        writer = SummaryWriter(log_dir=fold_log_dir)
        best_auc = 0.0

        # (C) Epoch별 학습 + 검증 루프
        for epoch in range(cfg["training"]["epochs"]):
            #########################
            # (C-1) Training 단계
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
            # (C-2) Validation 단계
            #########################
            model.eval()
            all_labels_fold = []
            all_scores_fold = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1)[:, 1]  # fake 확률
                    all_labels_fold.extend(labels.cpu().numpy())
                    all_scores_fold.extend(probs.cpu().numpy())

            val_auc = compute_auc(np.array(all_labels_fold), np.array(all_scores_fold))
            val_eer = compute_eer(np.array(all_labels_fold), np.array(all_scores_fold))
            writer.add_scalar("AUC/val", val_auc, epoch)
            writer.add_scalar("EER/val", val_eer, epoch)

            print(
                f"[Fold {fold+1}/{k_folds} | Epoch {epoch+1}/{cfg['training']['epochs']}] "
                f"Train Loss: {avg_train_loss:.4f}  Val AUC: {val_auc:.4f}  Val EER: {val_eer:.4f}"
            )

            #########################
            # (C-3) 최적 모델 저장
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

    # 6) K-Fold 완료 후 Test 셋 평가
    print("\n=== K-Fold Training Complete ===")
    print("Evaluating on the Test set using Fold 5's best model...")

    # (D) Test 데이터셋 로드
    test_subset = Subset(full_dataset, test_idx)
    test_loader  = DataLoader(
        test_subset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )
    print(f"Test samples: {len(test_subset)}")

    # (E) 마지막 Fold(best.pth) 모델 로드
    last_ckpt_path = os.path.join(cfg["output_dir"], f"fold_{k_folds}", "best.pth")
    assert os.path.exists(last_ckpt_path), f"Checkpoint not found: {last_ckpt_path}"

    model = DeepfakeAdapter(cfg).to(device)
    ckpt = torch.load(last_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # (F) Test 셋 평가
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
    print(f"Final Test AUC: {test_auc:.4f}, Test EER: {test_eer:.4f}")


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
