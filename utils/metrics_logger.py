# utils/metrics_logger.py
"""
Matrix and Detailed Metrics Logger for RAPT-CLIP.

Provides formatted multi-label and single-label evaluation reports:
- Multi-label: Per-class AP, Positive Counts, Pred Counts, Precision, Recall, F1, Mean Probability,
  and Top/Bottom performing categories.
- Single-label: Formatted Confusion Matrix (counts & normalized), Per-class Recall/Precision/F1, WAR, UAR.
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix


def get_dataset_class_names(dataset_name, num_classes=None):
    """Returns human-readable class names for the specified dataset."""
    from models.Text import (
        class_names_5,
        class_names_7,
        class_names_8,
        class_names_emotic,
        class_names_ckplus,
        class_names_daisee,
        class_names_caer
    )
    if dataset_name == "EMOTIC":
        return class_names_emotic
    elif dataset_name == "RAER":
        if num_classes == 7:
            return class_names_7
        elif num_classes == 8:
            return class_names_8
        return [c.split(' (')[0] for c in class_names_5]
    elif dataset_name == "DAiSEE":
        return class_names_daisee
    elif dataset_name == "CK+":
        return class_names_ckplus
    elif dataset_name == "CAER" or dataset_name == "CAER-S":
        return class_names_caer
    else:
        if num_classes is not None:
            return [f"Class_{i}" for i in range(num_classes)]
        return None


def format_multilabel_matrix_report(targets, preds, class_names=None, threshold=0.5):
    """
    Generates a detailed per-class matrix report for multi-label datasets (e.g. EMOTIC).
    
    Args:
        targets: (N, C) numpy array or torch tensor of binary ground truths
        preds: (N, C) numpy array or torch tensor of sigmoid probabilities [0, 1]
        class_names: list of C string names
        threshold: float decision threshold (default: 0.5)
        
    Returns:
        report_str: Formatted multi-line string
        metrics_dict: Dictionary containing macro mAP, per-class AP, precisions, recalls, F1s
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
        
    N, C = targets.shape
    if class_names is None or len(class_names) != C:
        class_names = [f"Class_{i}" for i in range(C)]
        
    per_class_ap = []
    per_class_gt_pos = []
    per_class_pred_pos = []
    per_class_prec = []
    per_class_rec = []
    per_class_f1 = []
    per_class_mean_prob = []
    
    bin_preds = (preds >= threshold).astype(float)
    
    for c in range(C):
        y_true = targets[:, c]
        y_pred_prob = preds[:, c]
        y_pred_bin = bin_preds[:, c]
        
        # Average Precision
        if np.sum(y_true) > 0:
            ap = average_precision_score(y_true, y_pred_prob) * 100.0
        else:
            ap = 0.0
        per_class_ap.append(ap)
        
        # Support
        gt_pos = int(np.sum(y_true))
        pred_pos = int(np.sum(y_pred_bin))
        per_class_gt_pos.append(gt_pos)
        per_class_pred_pos.append(pred_pos)
        
        # Confusion metrics
        tp = np.sum((y_true == 1) & (y_pred_bin == 1))
        fp = np.sum((y_true == 0) & (y_pred_bin == 1))
        fn = np.sum((y_true == 1) & (y_pred_bin == 0))
        
        prec = (tp / (tp + fp + 1e-6)) * 100.0
        rec = (tp / (tp + fn + 1e-6)) * 100.0
        f1 = (2 * prec * rec / (prec + rec + 1e-6))
        
        per_class_prec.append(prec)
        per_class_rec.append(rec)
        per_class_f1.append(f1)
        per_class_mean_prob.append(float(np.mean(y_pred_prob)))
        
    macro_map = np.mean(per_class_ap)
    macro_prec = np.mean(per_class_prec)
    macro_rec = np.mean(per_class_rec)
    macro_f1 = np.mean(per_class_f1)
    
    # Sort classes by AP
    sorted_indices = np.argsort(per_class_ap)
    bottom_5 = [(class_names[i], per_class_ap[i]) for i in sorted_indices[:5]]
    top_5 = [(class_names[i], per_class_ap[i]) for i in sorted_indices[-5:][::-1]]
    
    # Build text report
    lines = []
    lines.append("=" * 112)
    lines.append(f"                           MULTI-LABEL CLASSIFICATION DETAILED MATRIX REPORT (N={N})")
    lines.append("=" * 112)
    lines.append(f"{'Class Name':<20} | {'AP (%)':>8} | {'GT Pos (%)':>14} | {'Pred Pos (%)':>14} | {'Prec (%)':>9} | {'Rec (%)':>8} | {'F1 (%)':>8} | {'Mean Prob':>9}")
    lines.append("-" * 112)
    
    for c in range(C):
        c_name = class_names[c]
        ap = per_class_ap[c]
        gt_cnt = per_class_gt_pos[c]
        gt_pct = (gt_cnt / N) * 100.0
        pred_cnt = per_class_pred_pos[c]
        pred_pct = (pred_cnt / N) * 100.0
        prec = per_class_prec[c]
        rec = per_class_rec[c]
        f1 = per_class_f1[c]
        mean_p = per_class_mean_prob[c]
        
        gt_str = f"{gt_cnt:>4} ({gt_pct:>4.1f}%)"
        pred_str = f"{pred_cnt:>4} ({pred_pct:>4.1f}%)"
        
        lines.append(f"{c_name:<20} | {ap:>7.2f}% | {gt_str:>14} | {pred_str:>14} | {prec:>8.1f}% | {rec:>7.1f}% | {f1:>7.1f}% | {mean_p:>9.4f}")
        
    lines.append("-" * 112)
    lines.append(f"{'Macro Average':<20} | {macro_map:>7.2f}% | {'':>14} | {'':>14} | {macro_prec:>8.1f}% | {macro_rec:>7.1f}% | {macro_f1:>7.1f}% |")
    lines.append("=" * 112)
    
    top_str = ", ".join([f"{name} ({val:.1f}%)" for name, val in top_5])
    bottom_str = ", ".join([f"{name} ({val:.1f}%)" for name, val in bottom_5])
    lines.append(f"Top-5 Strongest Classes : {top_str}")
    lines.append(f"Bottom-5 Weakest Classes: {bottom_str}")
    lines.append("=" * 112)
    
    report_str = "\n".join(lines)
    metrics_dict = {
        'macro_map': macro_map,
        'per_class_ap': dict(zip(class_names, per_class_ap)),
        'per_class_prec': dict(zip(class_names, per_class_prec)),
        'per_class_rec': dict(zip(class_names, per_class_rec)),
        'per_class_f1': dict(zip(class_names, per_class_f1)),
        'top_5': top_5,
        'bottom_5': bottom_5
    }
    return report_str, metrics_dict


def format_confusion_matrix_report(targets, preds, class_names=None):
    """
    Generates a formatted confusion matrix and per-class precision/recall table for single-label datasets.
    
    Args:
        targets: (N,) ground truth class indices
        preds: (N,) predicted class indices
        class_names: list of class names
        
    Returns:
        report_str: Formatted multi-line string with ASCII confusion matrix
        metrics_dict: Dictionary containing WAR, UAR, per-class accuracies, CM
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
        
    C = len(class_names) if class_names is not None else int(max(np.max(targets), np.max(preds)) + 1)
    if class_names is None or len(class_names) != C:
        class_names = [f"C{i}" for i in range(C)]
        
    cm = confusion_matrix(targets, preds, labels=range(C))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6) * 100.0
    
    class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6) * 100.0
    precisions = cm.diagonal() / (cm.sum(axis=0) + 1e-6) * 100.0
    uar = float(np.nanmean(class_acc))
    war = float(np.sum(cm.diagonal()) / np.sum(cm) * 100.0)
    
    # Abbreviated header names if too long
    col_headers = [c[:8] for c in class_names]
    col_w = max(9, max(len(h) for h in col_headers) + 2)
    
    lines = []
    lines.append("=" * (24 + col_w * C + 16))
    lines.append(f"                         CONFUSION MATRIX REPORT (WAR={war:.2f}%, UAR={uar:.2f}%)")
    lines.append("=" * (24 + col_w * C + 16))
    
    # Header row
    prefix_col = "True \\ Pred"
    header_str = f"{prefix_col:<22} | " + " | ".join([f"{h:^{col_w}}" for h in col_headers]) + f" | {'Recall (Acc)':^14}"
    lines.append(header_str)
    lines.append("-" * len(header_str))
    
    for i, name in enumerate(class_names):
        row_counts = [f"{cm[i, j]:^{col_w}}" for j in range(C)]
        row_str = f"{name:<22} | " + " | ".join(row_counts) + f" | {class_acc[i]:>11.2f}%"
        lines.append(row_str)
        
    lines.append("-" * len(header_str))
    # Precision row
    prec_strs = [f"{precisions[j]:^{col_w}.1f}%" for j in range(C)]
    lines.append(f"{'Precision (%)':<22} | " + " | ".join(prec_strs) + f" | WAR: {war:>8.2f}%")
    
    # Support row
    col_sums = [f"{np.sum(cm[:, j]):^{col_w}}" for j in range(C)]
    lines.append(f"{'Pred Count':<22} | " + " | ".join(col_sums) + f" | UAR: {uar:>8.2f}%")
    lines.append("=" * (24 + col_w * C + 16))
    
    report_str = "\n".join(lines)
    metrics_dict = {
        'war': war,
        'uar': uar,
        'per_class_acc': dict(zip(class_names, class_acc)),
        'confusion_matrix': cm,
        'normalized_cm': cm_norm
    }
    return report_str, metrics_dict
