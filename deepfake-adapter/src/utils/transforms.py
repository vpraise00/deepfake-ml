# src/utils/transforms.py

from torchvision import transforms

def get_ffpp_transforms(input_size: int):
    """
    FaceForensics++ 전용 이미지 전처리 파이프라인을 반환합니다.
    - Resize → ToTensor → Normalize(ImageNet 통계)
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
