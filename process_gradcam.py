import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

def generate_gradcam(input_tensor, model, target_layer):
    """
    주어진 입력 텐서에 대해 GradCAM 히트맵을 계산하는 함수.
    
    Parameters:
        input_tensor (torch.Tensor): 전처리된 이미지 텐서, shape=(1, C, H, W)
        model: 학습된 딥러닝 모델
        target_layer: GradCAM 적용 대상인 모델의 convolutional layer
        
    Returns:
        gradcam (np.array): [0, 1] 범위로 정규화된 GradCAM 히트맵 (크기: 224x224)
    """
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook 등록
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward Pass
    output = model(input_tensor)
    model.zero_grad()

    # 딥페이크 확률(클래스 0)을 대상으로 역전파 진행
    target = output[:, 0]
    target.backward()

    # Hook 제거
    forward_handle.remove()
    backward_handle.remove()

    # 수집된 activation 및 gradient 추출 (배치 차원 제거)
    activation = activations[0].squeeze(0).cpu().detach().numpy()
    gradient = gradients[0].squeeze(0).cpu().detach().numpy()

    # Global Average Pooling을 통해 가중치 계산
    weights = np.mean(gradient, axis=(1, 2))
    gradcam = np.maximum(np.sum(weights[:, None, None] * activation, axis=0), 0)

    # 히트맵 정규화 및 크기 조정 (224x224)
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
    gradcam = cv2.resize(gradcam, (224, 224))

    return gradcam

def apply_gradcam_overlay(image_path, heatmap):
    """
    원본 이미지에 GradCAM 히트맵을 컬러 오버레이하여 반환하는 함수.
    
    Parameters:
        image_path (str): 원본 이미지 파일 경로
        heatmap (np.array): [0, 1] 범위의 GradCAM 히트맵
        
    Returns:
        overlay (np.array): GradCAM 히트맵이 적용된 오버레이 이미지
    """
    # Windows에서 한글 경로 문제를 해결하기 위해 np.fromfile와 cv2.imdecode 사용
    try:
        img_array = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}") from e

    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

    # 원본 이미지 크기에 맞게 히트맵 재조정
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay


#지워도 됨.
def process_gradcam_for_folder(input_folder, model, target_layer, output_folder="gradcam", transform=None, device=None):
    """
    지정한 폴더 내의 모든 이미지에 대해 GradCAM 처리를 수행하여,
    오버레이 이미지를 output_folder에 저장하는 함수.
    
    Parameters:
        input_folder (str): 처리할 이미지가 있는 폴더 경로
        model: 학습된 딥러닝 모델
        target_layer: GradCAM을 적용할 모델의 convolutional layer
        output_folder (str): GradCAM 결과 이미지 저장 폴더 (기본값: "gradcam")
        transform: 이미지 전처리 transform (기본값: Resize(224,224) + ToTensor)
        device: torch.device (기본값: CUDA 사용 가능 시 cuda, 아니면 cpu)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 입력 폴더 내의 jpg 이미지 목록 (소문자 확장자 포함)
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(".jpg")])
    
    for image_file in tqdm(image_files, desc="GradCAM 처리 중"):
        image_path = os.path.join(input_folder, image_file)
        try:
            # PIL을 사용해 이미지 열기 및 전처리
            img_pil = Image.open(image_path).convert("RGB")
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            # GradCAM 히트맵 계산
            heatmap = generate_gradcam(input_tensor, model, target_layer)
            
            # 원본 이미지에 GradCAM 오버레이 적용
            gradcam_overlay = apply_gradcam_overlay(image_path, heatmap)
            
            # 결과 파일명: 원본 파일명에 _gradcam 접미어 추가
            base_name, ext = os.path.splitext(image_file)
            output_path = os.path.join(output_folder, f"{base_name}_gradcam{ext}")
            cv2.imwrite(output_path, gradcam_overlay)
        except Exception as e:
            print(f"{image_path} 처리 중 에러 발생: {e}")
