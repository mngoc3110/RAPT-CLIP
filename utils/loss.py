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
