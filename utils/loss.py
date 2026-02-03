# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal

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