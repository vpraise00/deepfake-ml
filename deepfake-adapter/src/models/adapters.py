# src/models/adapters.py

import torch
import torch.nn as nn


class GlobalAdapterBlock(nn.Module):
    """
    Global-aware Adapter Block (GAB)
    - 입력: CLS 토큰 임베딩 (B, D)
    - 내부적으로 작은 MLP (dim → adapter_dim → dim) + Residual + LayerNorm
    - 출력: (B, D)
    """
    def __init__(self, dim: int, adapter_dim: int):
        """
        Args:
            dim (int): 입력 CLS 토큰 임베딩 차원 (예: 768)
            adapter_dim (int): 어댑터 내부 MLP 차원 (예: 128)
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, adapter_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(adapter_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): CLS 토큰 임베딩, shape = (B, D)
        Returns:
            torch.Tensor: 보정된 CLS 임베딩, shape = (B, D)
        """
        residual = x
        out = self.fc1(x)        # (B, adapter_dim)
        out = self.relu(out)
        out = self.fc2(out)      # (B, D)
        # Residual 연결 후 LayerNorm
        return self.norm(out + residual)


class LocalAdapterBlock(nn.Module):
    """
    Local-aware Adapter Block (LAB)
    - 입력: 패치 임베딩 시퀀스 (B, N, D)
    - DepthwiseConv1d → MLP (dim → adapter_dim → dim) → Residual + LayerNorm
    - 출력: (B, N, D)
    """
    def __init__(self, dim: int, adapter_dim: int, num_patches: int):
        """
        Args:
            dim (int): 패치 임베딩 차원 (예: 768)
            adapter_dim (int): 어댑터 내부 MLP 차원 (예: 128)
            num_patches (int): 패치 개수 N (예: (224/16)^2 = 14*14 = 196)
        """
        super().__init__()
        # Depthwise 1D convolution: groups=dim 으로 채널별 분리
        self.dw_conv = nn.Conv1d(in_channels=dim,
                                 out_channels=dim,
                                 kernel_size=3,
                                 padding=1,
                                 groups=dim,
                                 bias=False)
        self.fc1 = nn.Linear(dim, adapter_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(adapter_dim, dim)
        self.norm = nn.LayerNorm(dim)

        # (num_patches 은 실제 연산에는 별도 사용되지 않으나 추후 참조 가능)
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): 패치 임베딩 시퀀스, shape = (B, N, D)
        Returns:
            torch.Tensor: 보정된 패치 임베딩 시퀀스, shape = (B, N, D)
        """
        # Residual 복사
        residual = x  # (B, N, D)

        # Conv1d expects (B, D, N)
        out = x.permute(0, 2, 1)              # (B, D, N)
        out = self.dw_conv(out)               # (B, D, N)
        out = out.permute(0, 2, 1)             # (B, N, D)

        # MLP
        out = self.fc1(out)                   # (B, N, adapter_dim)
        out = self.relu(out)
        out = self.fc2(out)                   # (B, N, D)

        # Residual 연결 후 LayerNorm (patch 차원별로 병렬 처리)
        out = self.norm(out + residual)       # LayerNorm은 마지막 차원 D에 적용
        return out
