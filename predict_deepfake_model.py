import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm.models.xception as Xception
from PIL import Image
import os
import json
from tqdm import tqdm
import numpy as np
import cv2

# process_gradcam.py에서 GradCAM 관련 함수들을 import
from process_gradcam import generate_gradcam, apply_gradcam_overlay

# ✅ 모델 불러오기
MODEL_PATH = "model/CelebDF_model_20_epochs_99acc.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 로드
# 모델 체크포인트를 불러와 모델 구조와 가중치를 설정하고, 평가 모드로 전환합니다.
torch.serialization.add_safe_globals([Xception])
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model = checkpoint['model']
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device, memory_format=torch.channels_last)
model.eval()

# ✅ 마지막 Convolutional Layer 설정 (XceptionNet 구조에 맞게 수정)
# GradCAM을 적용할 대상 layer를 지정합니다.
target_layer = model.conv1  # 🚩 모델 구조에 따라 변경 필요

# ✅ 이미지 전처리 (224x224 크기로 변환)
# PIL 이미지 데이터를 224x224 크기로 변환 후 Tensor로 변경합니다.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ 이미지 로드 (배치 단위)
def load_images_in_batches(image_paths, batch_size=16):
    """
    주어진 이미지 경로 리스트를 배치 단위로 읽어와 전처리한 텐서와 이미지 경로 리스트를 반환하는 generator 함수입니다.
    
    Parameters:
        image_paths (list): 이미지 파일 경로들의 리스트.
        batch_size (int): 한 번에 처리할 이미지 수.
        
    Yields:
        tuple: (batch_images, batch_paths)
            - batch_images: 전처리된 이미지 텐서 배치.
            - batch_paths: 해당 이미지 파일들의 경로 리스트.
    """
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        # 각 이미지 파일을 열고, RGB 모드로 변환 후 전처리(transform)를 적용합니다.
        images = [transform(Image.open(img_path).convert("RGB")) for img_path in batch_paths]
        yield torch.stack(images).to(device), batch_paths

# ✅ 예측 함수 (딥페이크 확률 계산)
def predict(images_tensor):
    """
    주어진 이미지 텐서 배치에 대해 모델을 사용하여 딥페이크 확률을 예측합니다.
    
    Parameters:
        images_tensor (torch.Tensor): 전처리된 이미지 텐서 배치.
        
    Returns:
        numpy.ndarray: 각 이미지에 대한 딥페이크 확률(0~1 사이 값).
    """
    with torch.no_grad():
        outputs = model(images_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()
    return probabilities

# ✅ 전체 프레임 처리 (각 프레임에 대해 예측 및 GradCAM 적용 후 결과를 JSON 파일로 저장)
def process_all_frames(root_folder, batch_size=16, use_gradcam=False):
    """
    지정한 폴더 내의 모든 이미지 파일(.jpg)에 대해 딥페이크 확률 예측을 수행하고,
    선택적으로 GradCAM을 적용하여 결과 이미지를 저장한 후, 예측 결과(확률 및 GradCAM 이미지 경로)를 JSON 파일로 저장합니다.
    
    Parameters:
        root_folder (str): 루트 폴더(ml).
        output_file (str): 예측 결과를 저장할 JSON 파일 경로.
        batch_size (int): 배치 처리 시 한 번에 읽어올 이미지 수.
        use_gradcam (bool): True이면 GradCAM을 적용하여 결과 이미지 저장.
    """
    # 입력 폴더 내의 모든 jpg 파일 경로를 정렬하여 리스트로 만듭니다.
    frames_dir = os.path.join(root_folder, "frames")
    #print("frames_dir 절대 경로:", os.path.abspath(frames_dir))
    #print("frames_dir 파일 목록:", os.listdir(frames_dir))
    image_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if not image_files:
        print("❌ 이미지 파일이 없습니다.")
        return


    results = []

    # GradCAM 적용 시, 결과 이미지를 저장할 "gradcam" 폴더(루트 경로에 생성)를 준비합니다.
    if use_gradcam:
        gradcam_folder = os.path.join(root_folder, "gradcam")
        os.makedirs(gradcam_folder, exist_ok=True)



    # 이미지 배치 처리
    for batch_images, batch_paths in tqdm(load_images_in_batches(image_files, batch_size),
                                            total=len(image_files) // batch_size + 1, desc="🔍 Processing"):
        # 각 배치에 대해 예측 수행
        probs = predict(batch_images)
        for path, img_tensor, prob in zip(batch_paths, batch_images, probs):
            frame_name = os.path.basename(path)
            results.append(float(prob))
            # result_data = {"probability": float(prob)}

            # GradCAM 적용 여부
            if use_gradcam:
                # generate_gradcam 함수는 입력 텐서가 (1, C, H, W) 형태를 요구하므로 unsqueeze 적용
                heatmap = generate_gradcam(img_tensor.unsqueeze(0), model, target_layer)
                # 원본 이미지와 heatmap을 이용해 GradCAM 오버레이 이미지 생성
                gradcam_image = apply_gradcam_overlay(path, heatmap)

                # gradcam 결과는 루트 폴더의 "gradcam" 폴더에 저장 (원본 파일명에 _gradcam 접미어 추가)
                base_name, ext = os.path.splitext(frame_name)
                gradcam_filename = f"gradcam_{base_name}{ext}"
                gradcam_path = os.path.join(gradcam_folder, gradcam_filename)
                cv2.imwrite(gradcam_path, gradcam_image)

                # JSON 결과에 GradCAM 이미지 경로 추가
                # result_data["gradcam_path"] = gradcam_path

            # 각 이미지의 결과 데이터를 results 딕셔너리에 저장합니다.
            # results[frame_name] = result_data
    return results

    # 모든 결과를 JSON 파일로 저장합니다.
    # with open(output_file, "w") as f:
    #     json.dump(results, f, indent=4)
    # print(f"✅ 예측 결과 저장 완료: {output_file}")

# ✅ 실행 예시 (Main)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="딥페이크 프레임 분석기 (CelebDF_model_20_epochs_99acc 모델)")
    parser.add_argument("--input", type=str, required=True, help="프레임이 저장된 폴더 경로")
    parser.add_argument("--output", type=str, default="results/deepfake_results.json", help="결과 저장 파일")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기 (기본값: 16)")
    parser.add_argument("--use_gradcam", action="store_true", help="GradCAM 적용 여부 (옵션 선택 시 적용)")

    args = parser.parse_args()
    process_all_frames(args.input, args.output, args.batch_size, args.use_gradcam)
