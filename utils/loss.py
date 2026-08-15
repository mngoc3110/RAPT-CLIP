# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
import numpy as np


class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()

    def forward(self, text_features):
        # Normalize features - Robust
        norm = text_features.norm(p=2, dim=-1, keepdim=True) + 1e-6
        text_features = text_features / norm
        
        # Calculate cosine similarity matrix
        similarity_matrix = torch.matmul(text_features, text_features.T)
        
        # Penalize off-diagonal elements
        loss = (similarity_matrix - torch.eye(text_features.shape[0], device=text_features.device)).pow(2).sum()
        
        return loss / (text_features.shape[0] * (text_features.shape[0] - 1))

class MILoss(nn.Module):
    def __init__(self, T=0.07):
        super(MILoss, self).__init__()
        self.T = T
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, learnable_text_features, hand_crafted_text_features):
        # Normalize features - Robust
        norm_learn = learnable_text_features.norm(p=2, dim=-1, keepdim=True) + 1e-6
        learnable_text_features = learnable_text_features / norm_learn
        
        norm_hand = hand_crafted_text_features.norm(p=2, dim=-1, keepdim=True) + 1e-6
        hand_crafted_text_features = hand_crafted_text_features / norm_hand
        
        # Calculate cosine similarity
        logits = torch.matmul(learnable_text_features, hand_crafted_text_features.T) / self.T
        
        # Create labels for positive pairs (diagonal elements)
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        # Calculate loss in both directions and average
        loss_l2h = self.criterion(logits, labels)
        loss_h2l = self.criterion(logits.T, labels)
        
        return (loss_l2h + loss_h2l) / 2

class LSR2(nn.Module):
    def __init__(self, e=0.1, label_mode='class_descriptor', reduction='mean'):
        super().__init__()
        self.epsilon = e
        self.reduction = reduction

    def forward(self, preds, target):
        """
        preds: (B, C) Logits
        target: (B) LongTensor of labels
        """
        n_classes = preds.size(1)
        log_preds = F.log_softmax(preds, dim=1)
        
        # Compute standard cross entropy part (for the true label)
        loss = -log_preds.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        
        # Compute smoothing part (average of all classes)
        loss = (1 - self.epsilon) * loss + self.epsilon * (-log_preds.mean(dim=1))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class BlvLoss(nn.Module):
    def __init__(self, cls_num_list, sigma=4, loss_name='BlvLoss'):
        super(BlvLoss, self).__init__()
        cls_list = torch.cuda.FloatTensor(cls_num_list)
        frequency_list = torch.log(cls_list)
        self.frequency_list = torch.log(sum(cls_list)) - frequency_list
        self.reduction = 'mean'
        self.sampler = normal.Normal(0, sigma)
        self._loss_name = loss_name

    def forward(self, pred, target):
        viariation = self.sampler.sample(pred.shape).clamp(-1, 1).to(pred.device)
        pred = pred + (viariation.abs() / self.frequency_list.max() * self.frequency_list)
        loss = F.cross_entropy(pred, target, reduction='none')

        return loss.mean()

class MoCoRankLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(MoCoRankLoss, self).__init__()
        self.temperature = temperature

    def forward(self, video_features, text_features, target, queue):
        """
        video_features: (B, D)
        text_features: (C, D)
        target: (B)
        queue: (D, K)
        """
        batch_size = video_features.shape[0]
        
        # 1. Positive Logits: Video vs Correct Text Prototype (B, 1)
        # Select the text prototype for each sample in the batch
        pos_text_prototypes = text_features[target] # (B, D)
        l_pos = torch.einsum('bd,bd->b', [video_features, pos_text_prototypes]).unsqueeze(-1) # (B, 1)
        
        # 2. Negative Logits: Video vs Queue (B, K)
        l_neg = torch.matmul(video_features, queue) # (B, K)
        
        # 3. Combine logits
        logits = torch.cat([l_pos, l_neg], dim=1) # (B, 1+K)
        logits /= self.temperature
        
        # 4. Labels: The positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=video_features.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.8, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        # Keep m_list as a CPU tensor initially, move to device in forward
        self.register_buffer('m_list', torch.FloatTensor(m_list)) 
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), True)
        
        index_float = index.to(dtype=x.dtype, device=x.device)
        
        # Ensure m_list is on the same device as input x
        m_list = self.m_list.to(dtype=x.dtype, device=x.device)
        
        batch_m = torch.matmul(m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

class SemanticLDLLoss(nn.Module):
    def __init__(self, temperature=1.0, target_temperature=0.1):
        super(SemanticLDLLoss, self).__init__()
        self.temperature = temperature
        self.target_temperature = target_temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, target, text_features):
        """
        logits: (B, C) - Video-Text similarities
        target: (B) - Ground truth indices
        text_features: (C, D) - Embeddings of class prompts
        """
        # 1. Compute Semantic Similarity between classes based on Text Features
        # Ensure features are normalized for cosine similarity - Robust
        norm_text = text_features.norm(p=2, dim=-1, keepdim=True) + 1e-6
        text_features = text_features / norm_text
        
        sim_matrix = torch.matmul(text_features, text_features.T) # (C, C)
        
        # 2. Create Soft Target Distributions
        # For each sample, the target distribution is the row in sim_matrix corresponding to the GT label
        soft_targets = sim_matrix[target] # (B, C)
        
        # Apply softmax to turn similarities into a valid probability distribution
        # Use a lower temperature to sharpen the targets (fix for high similarity prompts)
        soft_targets = F.softmax(soft_targets / self.target_temperature, dim=1)
        
        # 3. Compute Prediction Log-Probabilities
        # Use the model's temperature (or provided temp) for predictions
        log_probs = F.log_softmax(logits / self.temperature, dim=1)
        
        # 4. KL Divergence Loss
        loss = self.kl_div(log_probs, soft_targets)
        return loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        # Cast to float32 to prevent NaN with AMP (float16)
        x = x.float()
        y = y.float()

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class MaskedAsymmetricLoss(nn.Module):
    """
    Masked Asymmetric Loss for Multi-Label Classification.
    
    Extends ASL by randomly masking a portion of negative labels during training.
    This addresses the severe positive/negative imbalance in datasets like EMOTIC
    where each image has ~2 positive labels out of 26 (i.e., 24 negatives).
    
    The masking reduces the dominance of easy negatives, allowing the model to 
    focus on learning the positive labels and hard negatives.
    
    Args:
        gamma_neg (float): Focusing parameter for negative samples. Default: 4.
        gamma_pos (float): Focusing parameter for positive samples. Default: 0.
        clip (float): Asymmetric clipping threshold. Default: 0.05.
        mask_ratio (float): Ratio of negative labels to randomly mask. Default: 0.3.
        eps (float): Small value for numerical stability. Default: 1e-8.
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, mask_ratio=0.3, 
                 eps=1e-8, disable_torch_grad_focal_loss=True):
        super(MaskedAsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.mask_ratio = mask_ratio
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x: (B, C) raw logits
            y: (B, C) multi-hot target labels (0 or 1)
        """
        # Cast to float32 to prevent NaN with AMP (float16)
        x = x.float()
        y = y.float()

        # ===== Random Negative Masking =====
        # Create mask: keep all positives (y==1), randomly keep (1-mask_ratio) of negatives
        if self.training:
            neg_mask = (y == 0).float()
            # For each negative position, keep it with probability (1 - mask_ratio)
            keep_prob = 1.0 - self.mask_ratio * neg_mask
            random_mask = torch.bernoulli(keep_prob)  # (B, C) binary mask
        else:
            random_mask = torch.ones_like(y)  # No masking during eval

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Apply random mask (zeroes out masked negative positions)
        loss = loss * random_mask

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        # Normalize by number of unmasked positions (not total) for stable gradients
        num_unmasked = random_mask.sum().clamp(min=1.0)
        return -loss.sum() / num_unmasked * y.shape[1]  # Scale back to comparable magnitude

