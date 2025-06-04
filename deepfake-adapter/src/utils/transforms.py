# src/utils/transforms.py (수정본)

import torchvision.transforms as T

def get_ffpp_transforms(input_size: int):
    """
    학습/검증용 데이터 전처리 함수를 두 개의 Compose 객체로 반환합니다.
    
    Returns:
      - train_transform: 학습 시 적용할 증강(강한 Augmentation)
      - val_transform: 검증/테스트 시 적용할 최소 전처리
    """
    # --- 학습용 증강 ---
    train_transform = T.Compose([
        # 다양한 크롭 → 모델이 여러 스케일에 강건해짐
        T.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
        # 좌우 뒤집기 (확률 0.5)
        T.RandomHorizontalFlip(p=0.5),
        # 밝기·대비·채도·색조 랜덤 변경
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        # 확률적으로 가우시안 블러
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
        # 텐서 변환 + 정규화
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # --- 검증/테스트용 전처리 ---
    val_transform = T.Compose([
        # Resize → CenterCrop (예: input_size*1.143 → input_size)
        T.Resize(int(input_size * 1.143)),  
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform
