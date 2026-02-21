
import torch
from models.clip import clip
import numpy as np
import torch.nn.functional as F

# Import the prompts directly to avoid dependency hell with the whole project
prompt_ensemble_5 = [
    [   # Neutral
        "A photo of a student with a neutral expression.",
        "A photo of a student sitting still and watching the lecture.",
        "A photo of a student with a calm face and neutral body posture."
    ],
    [
        # Enjoyment
        "A photo of a student showing enjoyment while learning.",
        "A photo of a student with a happy face and a slight smile.",
        "A photo of a student who looks engaged and interested in the lesson."
    ],
    [
        # Confusion
        "A photo of a student who is confused.",
        "A photo of a student with a puzzled look and furrowed eyebrows.",
        "A photo of a student staring at the material as if trying to understand."
    ],
    [
        # Fatigue
        "A photo of a student who appears fatigued or sleepy.",
        "A photo of a student with drooping eyelids or yawning.",
        "A photo of a student showing low energy with a lowered head."
    ],
    [
        # Distraction
        "A photo of a student who is distracted from learning.",
        "A photo of a student looking away from the lesson or checking a phone.",
        "A photo of a student with a wandering gaze and unfocused eyes."
    ]
]

class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

def check_similarity():
    device = "cpu"
    # if torch.backends.mps.is_available():
    #     device = "mps"
    # elif torch.cuda.is_available():
    #     device = "cuda"
    
    print(f"Loading CLIP model on {device}...")
    model, _ = clip.load("ViT-B/16", device=device)
    
    print("Computing embeddings for prompt ensemble...")
    
    # Flatten the ensemble for processing, but we need to average them per class later
    # Or actually, the Trainer averages the *features* of the learnable prompts. 
    # Here we are simulating the "Static" or "Hand-Crafted" similarity if we were using fixed prompts,
    # OR the starting point of learnable prompts. 
    # Let's compute the similarity of the *averaged* embeddings for each class,
    # as this represents the "class prototype" in text space.
    
    class_embeddings = []
    
    with torch.no_grad():
        for i, prompts in enumerate(prompt_ensemble_5):
            tokenized = clip.tokenize(prompts).to(device)
            features = model.encode_text(tokenized)
            features = F.normalize(features, p=2, dim=-1)
            
            # Average features for this class
            class_embedding = features.mean(dim=0)
            class_embedding = F.normalize(class_embedding, p=2, dim=-1) # Re-normalize
            class_embeddings.append(class_embedding)
            
    # Stack into a matrix (C, D)
    text_features = torch.stack(class_embeddings)
    
    # Compute Similarity Matrix (C, C)
    sim_matrix = torch.matmul(text_features, text_features.T)
    
    print("\n" + "="*60)
    print("      SEMANTIC SIMILARITY MATRIX (CLIP ViT-B/16)")
    print("="*60)
    
    # Print Header
    header = f"{'':<12} | " + " | ".join([f"{name[:8]:<8}" for name in class_names])
    print(header)
    print("-" * len(header))
    
    # Print Rows
    for i, row in enumerate(sim_matrix):
        row_str = f"{class_names[i]:<12} | " + " | ".join([f"{val.item():.4f}" for val in row])
        print(row_str)
        
    print("="*60)
    
    # Analyze for LDL validity
    print("\nANALYSIS FOR LDL LOSS:")
    print("LDL assumes that if Class A and Class B are semantically similar,")
    print("an image of Class A should have a non-zero probability of being Class B.")
    
    threshold_high = 0.90
    threshold_low = 0.50
    
    problems = []
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            sim = sim_matrix[i, j].item()
            if sim > threshold_high:
                problems.append(f"WARNING: {class_names[i]} and {class_names[j]} are VERY similar ({sim:.4f}). Model might confuse them.")

    if not problems:
        print("Similarity distribution looks reasonable (no extremely high off-diagonal correlations).")
    else:
        for p in problems:
            print(p)

    # Show what the Soft Target would look like for 'Neutral' with T=1.0 and T=0.1
    print("\nExample Soft Targets (for Class 'Neutrality'):")
    
    def print_soft_target(temp):
        target_dist = F.softmax(sim_matrix[0] / temp, dim=0)
        print(f"\nTemperature = {temp}:")
        for k, val in enumerate(target_dist):
            print(f"  {class_names[k]}: {val.item():.4f}")

    print_soft_target(1.0)
    print_soft_target(0.1)

if __name__ == "__main__":
    check_similarity()
