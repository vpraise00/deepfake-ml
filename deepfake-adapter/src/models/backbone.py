# src/models/backbone.py

import torch
import torch.nn as nn
import timm


class ViTBackbone(nn.Module):
    """
    Vision Transformer 백본을 불러와서
    - CLS 토큰 임베딩 (B, D)
    - Patch 시퀀스     (B, N, D)
    두 가지를 모두 반환합니다.

    timm.create_model(..., pretrained=True)을 이용하고,
    forward_features(x) 호출로 hidden_states를 얻습니다.
    """

    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
        """
        Args:
            model_name (str): timm 라이브러리에서 사용할 ViT 모델 이름
            pretrained (bool): 사전 학습된 가중치 사용 여부
        """
        super().__init__()
        # timm에서 ViT 모델 생성
        # pretrained=True → ImageNet 가중치 로드
        self.model = timm.create_model(model_name, pretrained=pretrained)

        # timm ViT의 classifier head는 forward_features([B,3,H,W]) 이후
        # .head (nn.Linear) 를 거쳐 logits을 출력합니다.
        # 우리는 CLS 토큰과 패치 임베딩만 필요하므로
        # classifier(head) 레이어를 그대로 두되, forward에서 사용하지 않습니다.

        # 백본의 임베딩 차원(D)을 가져오기 위해
        # self.model.embed_dim 또는 self.model.num_features 사용
        try:
            self.dim = self.model.embed_dim
        except AttributeError:
            # 어떤 버전의 timm에서는 num_features로 저장
            self.dim = self.model.num_features

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): 입력 이미지 배치, 크기 [B, 3, H, W]

        Returns:
            cls_token (torch.Tensor): CLS 토큰 임베딩, 크기 [B, D]
            patch_tokens (torch.Tensor): 패치 시퀀스, 크기 [B, N, D]
                (N = number_of_patches = (H/patch_size) * (W/patch_size))
        """
        # timm ViT 모델에서 hidden_states를 얻으려면 forward_features() 사용
        # 대부분의 timm ViT 구현체에서 forward_features(x)는:
        #   - patch embedding → positional embedding → transformer layers → 
        #     LayerNorm → returns (B, N+1, D) 텐서 (CLS 포함)
        # 그러므로 아래처럼 호출하여 CLS+패치 시퀀스를 받아옵니다.
        hidden_states = self.model.forward_features(x)  # [B, N+1, D]

        # 첫 번째 토큰([:,0,:])이 CLS 토큰
        cls_token = hidden_states[:, 0, :]              # [B, D]
        # 나머지 토큰([:,1:,:])이 패치 시퀀스
        patch_tokens = hidden_states[:, 1:, :]           # [B, N, D]

        return cls_token, patch_tokens
