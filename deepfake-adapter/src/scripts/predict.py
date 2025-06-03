# src/scripts/predict.py

import os
import json
import yaml
import torch
from torch.utils.data import DataLoader
from PIL import Image
import cv2

from src.models.deepfake_adapter import DeepfakeAdapter
from src.utils.transforms import get_ffpp_transforms


def extract_frames_from_video(video_path: str, save_dir: str):
    """
    OpenCV를 사용하여 비디오에서 모든 프레임을 추출해
    save_dir/frame_{:06d}.jpg 형태로 저장합니다.
    """
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        save_path = os.path.join(save_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(save_path, frame)
        frame_idx += 1
    cap.release()


def main(config_path: str,
         checkpoint_path: str,
         video_path: str,
         output_json: str):
    # 1) Config 로드
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 모델 로드
    model = DeepfakeAdapter(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3) 임시 폴더에 프레임 추출
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        # 기존 파일 삭제
        for fname in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, fname))
    else:
        os.makedirs(temp_dir)
    extract_frames_from_video(video_path, temp_dir)

    # 4) 추출된 프레임 경로 리스트 생성
    transform = get_ffpp_transforms(cfg["input_size"])
    frame_paths = sorted([
        os.path.join(temp_dir, fn)
        for fn in os.listdir(temp_dir)
        if fn.endswith(".jpg") or fn.endswith(".png")
    ])

    # 임시 Dataset 정의
    class TempFrameDataset(torch.utils.data.Dataset):
        def __init__(self, paths, transform):
            self.paths = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.paths[idx]

    temp_dataset = TempFrameDataset(frame_paths, transform)
    temp_loader = DataLoader(
        temp_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )

    # 5) 예측
    results = {}  # { "frame_path": fake_prob }
    with torch.no_grad():
        for imgs, paths in temp_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # fake 확률
            for p, prob in zip(paths, probs):
                results[p] = float(prob)

    # 6) JSON 저장
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Prediction results saved to {output_json}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict Deepfake probabilities for a video"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ffpp_c23.yaml",
        help="Path to the config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pth checkpoint"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/predictions/result.json",
        help="Path to save the JSON output"
    )
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.video, args.output_json)
