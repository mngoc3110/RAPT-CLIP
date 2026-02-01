# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal

class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection.
    A solution to address the problem of class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass.
        :param inputs: (tensor) an float tensor of shape (N, C), where C is the number of classes.
        :param targets: (tensor) an long tensor of shape (N,).
        :return: (tensor) focal loss.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()

    def forward(self, text_features):
        # Normalize features
        text_features = F.normalize(text_features, p=2, dim=-1)
        
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
        # Normalize features
        learnable_text_features = F.normalize(learnable_text_features, p=2, dim=-1)
        hand_crafted_text_features = F.normalize(hand_crafted_text_features, p=2, dim=-1)
        
        # Calculate cosine similarity
        logits = torch.matmul(learnable_text_features, hand_crafted_text_features.T) / self.T
        
        # Create labels for positive pairs (diagonal elements)
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        # Calculate loss in both directions and average
        loss_l2h = self.criterion(logits, labels)
        loss_h2l = self.criterion(logits.T, labels)
        
        return (loss_l2h + loss_h2l) / 2

class LSR2(nn.Module):
    def __init__(self,e,label_mode):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.e = e
        self.label_mode = label_mode

    def _one_hot(self, labels, classes, value=1):
        one_hot = torch.zeros(labels.size(0), classes)
        labels = labels.view(labels.size(0), -1)
        value_added = torch.Tensor(labels.size(0), 1).fill_(value)
        value_added = value_added.to(labels.device)
        one_hot = one_hot.to(labels.device)
        one_hot.scatter_add_(1, labels, value_added)
        return one_hot

    def _smooth_label(self, target, length, smooth_factor):
        one_hot = self._one_hot(target, length, value=1 - smooth_factor)
        mask = (one_hot==0)
        balance_weight = torch.tensor([0.065267810,0.817977729,1.035884371,0.388144355,0.19551041668]).to(one_hot.device)
        ex_weight = balance_weight.expand(one_hot.size(0),-1)
        resize_weight = ex_weight[mask].view(one_hot.size(0),-1)
        resize_weight /= resize_weight.sum(dim=1, keepdim=True)
        one_hot[mask] += (resize_weight*smooth_factor).view(-1)
        return one_hot.to(target.device)
    
    def forward(self, x, target):
        smoothed_target = self._smooth_label(target, x.size(1), self.e)
        x = self.log_softmax(x)
        loss = torch.sum(- x * smoothed_target, dim=1)
        return torch.mean(loss)

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

class SemanticLDLLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(SemanticLDLLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, target, text_features):
        """
        logits: (B, C) - Video-Text similarities
        target: (B) - Ground truth indices
        text_features: (C, D) - Embeddings of class prompts
        """
        # 1. Compute Semantic Similarity between classes based on Text Features
        # text_features is (C, D), normalized
        # sim_matrix: (C, C)
        sim_matrix = torch.matmul(text_features, text_features.T)
        
        # 2. Create Soft Target Distributions
        # For each sample, the target distribution is the row in sim_matrix corresponding to the GT label
        # (B, C)
        soft_targets = sim_matrix[target]
        
        # Normalize soft targets to be a valid probability distribution
        soft_targets = F.softmax(soft_targets / self.temperature, dim=1)
        
        # 3. Compute Prediction Log-Probabilities
        log_probs = F.log_softmax(logits / self.temperature, dim=1)
        
        # 4. KL Divergence Loss
        loss = self.kl_div(log_probs, soft_targets)
        return loss