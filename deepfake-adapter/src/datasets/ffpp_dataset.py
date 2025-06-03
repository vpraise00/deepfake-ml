# 반드시 이 코드로 교체해야 합니다.
# src/datasets/ffpp_dataset.py

import os
import random
from pathlib import Path

import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.utils.transforms import get_ffpp_transforms


class FFPPFrameDataset(Dataset):
    """
    FaceForensics++ 프레임 기반 데이터셋 (alltype 구조).

    - dataset_root/version/alltype/ 하위에 조작 타입(Deepfakes, FaceSwap, Face2Face, NeuralTextures, real) 폴더가 있음.
    - 각 조작 타입 폴더 안에는 video_id 별 하위 폴더가 있고, 그 안에 해당 비디오의 이미지(프레임)들이 있음.
    - Deepfakes, FaceSwap → fake (label=1)
      Face2Face, NeuralTextures, real → real (label=0)
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path (str): 예) "configs/ffpp_c23.yaml"
        """
        # 1) 설정 로드
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # “alltype” 폴더 아래 전체 이미지 로드
        self.dataset_root = Path(cfg["dataset_root"]) / cfg["version"] / "alltype"
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Directory not found: {self.dataset_root}")

        # 2) 전처리 파이프라인
        self.transform = get_ffpp_transforms(cfg["input_size"])

        # 3) 레이블 정의
        self.fake_types = {"Deepfakes", "FaceSwap"}
        self.real_types = {"Face2Face", "NeuralTextures", "real"}

        # 4) 이미지 경로 + 라벨 수집
        self.image_paths = []  # 프레임 이미지 파일 경로 목록 (Path 객체)
        self.labels = []       # 각 이미지의 레이블 (0: real, 1: fake)

        for manipulation in os.listdir(self.dataset_root):
            manipulation_dir = self.dataset_root / manipulation
            if not manipulation_dir.is_dir():
                continue

            # 조작 타입별 레이블 결정
            if manipulation in self.fake_types:
                label = 1
            elif manipulation in self.real_types:
                label = 0
            else:
                print(f"Warning: Unknown folder '{manipulation}', skipping.")
                continue

            # 각 조작 타입 폴더 아래 video_id별 폴더 순회
            for video_id in os.listdir(manipulation_dir):
                video_dir = manipulation_dir / video_id
                if not video_dir.is_dir():
                    continue

                # video_dir 안의 모든 .jpg/.png 파일 수집
                for ext in ("*.jpg", "*.png"):
                    for img_file in video_dir.glob(ext):
                        self.image_paths.append(img_file)
                        self.labels.append(label)

        # 5) 데이터 섞기
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): 인덱스
        Returns:
            image (Tensor): 전처리된 이미지 텐서
            label (int): 0 또는 1
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
