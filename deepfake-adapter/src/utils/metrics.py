# src/utils/metrics.py

import numpy as np
from sklearn import metrics

def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    ROC AUC를 계산하여 반환합니다.
    Args:
        labels:  정답 레이블 배열 (0, 1)
        scores: 모델이 출력한 positive 클래스(‘fake’) 확률 배열
    Returns:
        AUC 값 (float)
    """
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    return metrics.auc(fpr, tpr)

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    EER(Equal Error Rate)을 계산하여 반환합니다.
    Args:
        labels:  정답 레이블 배열 (0, 1)
        scores: 모델이 출력한 positive 클래스 확률 배열
    Returns:
        EER 값 (float)
    """
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # FPR과 FNR이 가장 가까운 지점의 인덱스를 찾는다
    idx_eer = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx_eer])

def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    정확도(Accuracy)를 계산하여 반환합니다.
    Args:
        preds:  예측 레이블 배열 (0, 1)
        labels: 정답 레이블 배열 (0, 1)
    Returns:
        accuracy 값 (float)
    """
    correct = (preds == labels).astype(int).sum()
    return float(correct) / len(labels)
