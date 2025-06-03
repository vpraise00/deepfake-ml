# src/models/deepfake_adapter.py

import torch
import torch.nn as nn

from src.models.backbone import ViTBackbone
from src.models.adapters import GlobalAdapterBlock, LocalAdapterBlock


class DeepfakeAdapter(nn.Module):
    """
    DeepFake-Adapter 모델
    - Vision Transformer 백본(ViTBackbone)에서 CLS 토큰과 패치 임베딩을 추출
    - GlobalAdapterBlock을 통해 CLS 토큰 보정
    - LocalAdapterBlock을 통해 패치 임베딩 보정
    - 보정된 CLS 토큰과 평균 풀링한 패치 임베딩을 합쳐 classification head로 전달
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg (dict): configs/ffpp_*.yaml에서 로드한 설정 데이터.
                cfg["model"]["backbone"]   : timm 모델 이름 (예: "vit_base_patch16_224")
                cfg["model"]["adapter_dim"] : 어댑터 내부 MLP 차원 (예: 128)
                cfg["model"]["num_classes"]: 최종 클래스 수 (real vs fake → 2)
                cfg["input_size"]           : 입력 이미지 해상도 (예: 224)
        """
        super().__init__()

        # 1) ViT 백본 초기화
        backbone_name = cfg["model"]["backbone"]
        self.backbone = ViTBackbone(model_name=backbone_name, pretrained=True)
        # 백본의 임베딩 차원 (CLS/패치 임베딩 모두 동일)
        dim = self.backbone.dim

        # 2) Dual-Level Adapter 블록 생성
        adapter_dim = cfg["model"]["adapter_dim"]
        # Global Adapter (CLS 전용)
        self.global_adapter = GlobalAdapterBlock(dim=dim, adapter_dim=adapter_dim)

        # 패치 개수 = (input_size / patch_size)^2
        # timm ViTPatch16 → patch_size = 16
        num_patches = (cfg["input_size"] // 16) ** 2
        # Local Adapter (패치 시퀀스 전용)
        self.local_adapter = LocalAdapterBlock(dim=dim, adapter_dim=adapter_dim, num_patches=num_patches)

        # 3) 분류기 head: LayerNorm → Linear
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, cfg["model"]["num_classes"])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 입력 이미지 배치, shape = [B, 3, H, W]

        Returns:
            torch.Tensor: [B, num_classes] logits
        """
        # 1) ViT 백본에서 CLS 토큰과 패치 임베딩을 얻음
        cls_token, patch_tokens = self.backbone(x)
        # cls_token:    [B, D]
        # patch_tokens: [B, N, D]

        # 2) Global Adapter: CLS 보정
        g_out = self.global_adapter(cls_token)  # [B, D]

        # 3) Local Adapter: 패치 시퀀스 보정
        l_out = self.local_adapter(patch_tokens)  # [B, N, D]
        # 패치별 임베딩을 평균 풀링하여 (B, D) 형태로 만듦
        l_out_avg = l_out.mean(dim=1)             # [B, D]

        # 4) 두 출력을 합쳐 classification head에 입력
        combined = g_out + l_out_avg              # [B, D]
        logits = self.classifier(combined)        # [B, num_classes]
        return logits
