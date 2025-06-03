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
    FaceForensics++ 프레임 기반 데이터셋.
    
    - 사전에 FF++ 영상을 프레임 단위로 추출해둔 디렉터리(예: data/processed/c23/train/Deepfakes/*.jpg) 구조에서
      이미지 파일과 라벨(real=0, fake=1)을 읽어와 Dataset을 구성합니다.
    - split 인자로 "train", "val", "test"를 받고, 각 split 폴더 아래에 있는 조작 종류(manipulation)별 서브폴더를 순회합니다.
    - 이미지 당 하나의 레이블을 반환하며, get_ffpp_transforms 함수를 사용해 Resize→ToTensor→Normalize 전처리를 적용합니다.
    """

    def __init__(self, config_path: str, split: str = "train"):
        """
        Args:
            config_path (str): YAML 설정 파일 경로 (예: "configs/ffpp_c23.yaml").
            split (str): 데이터셋 분할명. "train", "val", "test" 중 하나.
        """
        # 1) YAML 설정 로드
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.dataset_root = Path(cfg["dataset_root"]) / cfg["version"]
        self.split = split
        self.transform = get_ffpp_transforms(cfg["input_size"])

        # 2) split 디렉터리(예: /path/to/FaceForensicspp/c23/train) 확인
        split_dir = self.dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # 3) 이미지 경로와 라벨(0: real, 1: fake) 수집
        self.image_paths = []
        self.labels = []

        # manipulation 폴더들: real/, Deepfakes/, FaceSwap/ 등
        for manipulation in os.listdir(split_dir):
            class_dir = split_dir / manipulation
            if not class_dir.is_dir():
                continue

            # 'real' 폴더는 라벨 0, 나머지는 모두 가짜(fake) 라벨 1 처리
            label = 0 if manipulation.lower() == "real" else 1

            # 클래스 폴더 내 모든 .jpg 파일을 순회
            for img_file in class_dir.glob("*.jpg"):
                self.image_paths.append(img_file)
                self.labels.append(label)

        # 4) 데이터 섞기(shuffle)
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): 인덱스
        Returns:
            image_tensor (Tensor): 전처리가 적용된 이미지 텐서, 크기 [3×H×W]
            label (int): 0 또는 1
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # PIL.Image로 이미지를 읽고 RGB로 변환
        image = Image.open(img_path).convert("RGB")
        # 정의된 transform (Resize→ToTensor→Normalize) 적용
        image = self.transform(image)

        return image, label
