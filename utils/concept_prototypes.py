# concept_prototypes.py
"""
Concept Generation & Refinement (CGR) Module from PromptCAD (TCSVT 2026).

Leverages fine-grained LLM/hand-crafted emotion descriptions, encodes them
via pre-trained CLIP text encoder, and applies K-means clustering to extract
noise-free representative concept prototypes (centroids).
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from models.clip import clip


def extract_concept_prototypes(prompts_per_class, clip_model, num_clusters=5, device='cpu'):
    """
    Given a list of prompts per class, encodes each prompt through CLIP text encoder
    and extracts K-means cluster centroids as concept prototypes.
    
    Args:
        prompts_per_class (list of lists or list of strings): 
            Either [[p1, p2, ...], ...] for ensemble prompts or [p1, p2, ...]
        clip_model: Loaded CLIP model
        num_clusters (int): Number of concept clusters (Default: 5, as in PromptCAD)
        device (str): Computation device
        
    Returns:
        prototypes (torch.Tensor): Shape (num_classes, embed_dim) normalized concept prototype embeddings
    """
    clip_model.eval()
    prototypes_list = []
    
    with torch.no_grad():
        for class_idx, class_prompts in enumerate(prompts_per_class):
            if isinstance(class_prompts, str):
                class_prompts = [class_prompts]
                
            # Tokenize and encode all prompts for this class
            tokens = clip.tokenize(class_prompts, truncate=True).to(device)
            text_features = clip_model.encode_text(tokens).float()  # (K, D)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-6)
            
            feats_np = text_features.cpu().numpy()
            n_samples = feats_np.shape[0]
            
            k = min(num_clusters, n_samples)
            if k > 1:
                # K-means clustering over prompt embeddings
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(feats_np)
                # Average of centroids as overall class concept prototype
                centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
                proto = centroids.mean(dim=0)
            else:
                proto = text_features.mean(dim=0)
                
            proto = proto / (proto.norm(dim=-1, keepdim=True) + 1e-6)
            prototypes_list.append(proto)
            
    prototypes = torch.stack(prototypes_list, dim=0)  # (C, D)
    return prototypes


def get_dataset_concept_prototypes(dataset_name, clip_model, num_clusters=5, device='cpu'):
    """
    Helper function to get pre-clustered concept prototypes for supported datasets.
    """
    from models.Text import (
        prompt_ensemble_emotic,
        prompt_ensemble_5,
        class_descriptor_daisee,
        class_descriptor_ckplus
    )
    
    if dataset_name == "EMOTIC":
        prompts = prompt_ensemble_emotic
    elif dataset_name == "RAER":
        prompts = prompt_ensemble_5
    elif dataset_name == "DAiSEE":
        prompts = [[p] for p in class_descriptor_daisee]
    elif dataset_name == "CK+":
        prompts = [[p] for p in class_descriptor_ckplus]
    else:
        from models.Text import class_descriptor_7_only_face
        prompts = [[p] for p in class_descriptor_7_only_face]
        
    return extract_concept_prototypes(prompts, clip_model, num_clusters=num_clusters, device=device)
