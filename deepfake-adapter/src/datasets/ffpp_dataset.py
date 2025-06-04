# src/datasets/ffpp_dataset.py

import os
import random
from pathlib import Path

import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset


class FFPPFrameDataset(Dataset):
    """
    FaceForensics++ 프레임 기반 데이터셋 (alltype 구조) with:
      - frame_interval: 프레임 샘플링 비율
      - transform: torchvision transform (학습/검증용 증강)
      - indices: 전체 이미지 리스트 중 사용할 인덱스 목록 (K-Fold 등에서 넘어옴)
    """

    def __init__(
        self,
        config_path: str,
        frame_interval: int = 1,
        transform=None,
        indices: list = None,
    ):
        """
        Args:
            config_path (str): 예) "configs/ffpp_c23.yaml"
            frame_interval (int): 영상 폴더 내에서 매 interval간격으로만 프레임을 샘플링.
                                  (예: interval=5 → 5프레임당 1장씩)
            transform: torchvision.transforms.Compose 객체. 적용할 전처리/증강.
            indices (list[int], optional): 전체 샘플 중 사용할 인덱스 리스트.
                                           None일 경우 전체 샘플을 사용.
        """
        super().__init__()
        # 1) 설정 로드
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # alltype 폴더 경로
        self.dataset_root = Path(cfg["dataset_root"]) / cfg["version"] / "alltype"
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Directory not found: {self.dataset_root}")

        self.transform = transform
        self.frame_interval = frame_interval

        # 2) 레이블 정의
        self.fake_types = {"Deepfakes", "FaceSwap"}
        self.real_types = {"Face2Face", "NeuralTextures", "real"}

        # 3) 이미지 경로 + 라벨 수집 (샘플링 적용)
        self.image_paths = []
        self.labels = []

        for manipulation in os.listdir(self.dataset_root):
            manipulation_dir = self.dataset_root / manipulation
            if not manipulation_dir.is_dir():
                continue

            if manipulation in self.fake_types:
                label = 1
            elif manipulation in self.real_types:
                label = 0
            else:
                print(f"Warning: Unknown folder '{manipulation}', skipping.")
                continue

            # - 영상 디렉터리(예: c23_frames/alltype/Deepfakes/video_001) 순회
            for video_id in os.listdir(manipulation_dir):
                video_dir = manipulation_dir / video_id
                if not video_dir.is_dir():
                    continue

                # 3-1) 모든 프레임 파일을 정렬하여 가져온 뒤
                img_files = sorted(video_dir.glob("*.jpg")) + sorted(video_dir.glob("*.png"))
                # 3-2) interval 단위로 샘플링
                sampled = img_files[:: self.frame_interval]
                for img_file in sampled:
                    self.image_paths.append(img_file)
                    self.labels.append(label)

        # 4) 전체(샘플링된) 이미지 리스트를 섞은 뒤
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

        # 5) indices가 주어졌다면 sub-selection
        if indices is not None:
            # indices는 이미 `self.image_paths` 정렬/셔플 후의 인덱스를 가리킴
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels      = [self.labels[i]      for i in indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
