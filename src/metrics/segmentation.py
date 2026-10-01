"""Fixed-threshold metrics. Undefined AUROC/recall are JSON null."""
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(probabilities, masks, threshold=0.5):
    p, y = np.asarray(probabilities), np.asarray(masks)
    if p.ndim != 4 or p.shape != y.shape or p.shape[1] != 1 or p.size == 0:
        raise ValueError('Expected matching nonempty N1HW arrays')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Probabilities must be finite and in [0,1]')
    if not np.isin(y, [0, 1]).all() or not 0 <= threshold <= 1:
        raise ValueError('Invalid binary masks or threshold')
    y = y.astype(bool)
    pred = p >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    actual_defects = y.reshape(len(y), -1).any(axis=1)
    predicted_defects = pred.reshape(len(y), -1).any(axis=1)
    return dict(pixel_auroc=float(roc_auc_score(y.ravel(), p.ravel())) if np.unique(y).size == 2 else None,
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.0,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.0,
                defect_recall=float(predicted_defects[actual_defects].mean()) if actual_defects.any() else None)
