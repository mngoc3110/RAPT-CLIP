import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score, confusion_matrix

from models.Generate_Model import GenerateModel
from models.clip import clip
from utils.builders import get_class_info, build_model

def get_args():
    parser = argparse.ArgumentParser(description='RAER Ablation Study Evaluation')
    parser.add_argument('--dataset', default='RAER', help='dataset name')
    parser.add_argument('--text-type', default='prompt_ensemble', help='Text type')
    parser.add_argument('--class-names-with-context', default='False', help='Class names with context')
    parser.add_argument('--temporal-layers', default=1, type=int)
    parser.add_argument('--contexts-number', default=8, type=int)
    parser.add_argument('--class-token-position', default='end', help='Class token position')
    parser.add_argument('--class-specific-contexts', default='True', help='Class specific contexts')
    parser.add_argument('--load-and-tune-prompt-learner', default='True', help='Load and tune prompt learner')
    parser.add_argument('--clip-path', default='ViT-B/16', help='CLIP model path')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--use-context', action='store_true', help='Use context branch')
    parser.add_argument('--fusion-type', default='cmaf', choices=['concat', 'cmaf', 'hierarchical'], help='Fusion type')
    parser.add_argument('--is-ensemble', action='store_true')
    parser.add_argument('--use-cocoop', action='store_true')
    parser.add_argument('--use-label-gcn', action='store_true')
    parser.add_argument('--drop_path_rate', default=0.0, type=float, help='Drop path rate for ViT')
    
    # Required parameters for data builder
    parser.add_argument('--use-weighted-sampler', action='store_true')
    
    args = parser.parse_args()
    return args

def evaluate_ablation(model, test_loader, device, class_names, condition="Original"):
    print(f"\n=> Evaluating condition: {condition}")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {condition}"):
            if len(batch) == 4:
                images_face, images_body, images_context, target = batch
                images_context = images_context.to(device)
            else:
                images_face, images_body, target = batch
                images_context = None

            images_face = images_face.to(device)
            images_body = images_body.to(device)
            target = target.to(device)
            
            # Apply Ablation Masking
            if condition == "Face Only":
                images_body = torch.zeros_like(images_body)
                if images_context is not None:
                    images_context = torch.zeros_like(images_context)
            elif condition == "Body Only":
                images_face = torch.zeros_like(images_face)
                if images_context is not None:
                    images_context = torch.zeros_like(images_context)
            elif condition == "Context Only":
                images_face = torch.zeros_like(images_face)
                images_body = torch.zeros_like(images_body)
            elif condition == "Face + Body":
                if images_context is not None:
                    images_context = torch.zeros_like(images_context)
            elif condition == "Original":
                pass
                
            # Forward pass
            output_tuple = model(images_face, images_body, images_context)
            output = output_tuple[0]
            
            _, predicted = torch.max(output.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    correct = (all_preds == all_targets).sum()
    war = correct / len(all_targets) * 100
    
    # Calculate UAR (macro recall)
    recalls = recall_score(all_targets, all_preds, average=None)
    uar = np.mean(recalls) * 100
    
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f"[{condition}] WAR: {war:.2f}%, UAR: {uar:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    return war, uar

def main():
    args = get_args()
    
    # Force context if requested
    args.use_context = True 
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    args.device = device
    args.freeze_image_encoder = False
    args.lr_image_encoder = 0
    args.use_ldl = False
    args.ldl_warmup = 5
    args.moco_dim = 512
    args.moco_k = 65536
    args.moco_m = 0.999
    args.moco_t = 0.07
    args.use_moco = False
    args.temperature = 0.07
    
    args.train_annotation = "RAER/annotation/train.txt"
    args.val_annotation = "RAER/annotation/val.txt"
    args.test_annotation = "RAER/annotation/test.txt"
    args.bounding_box_face = "RAER/bounding_box/face.json"
    args.bounding_box_body = "RAER/bounding_box/body.json"
    args.data_root = "RAER"
    args.crop_body = False
    
    class_names, input_text = get_class_info(args)
    model = build_model(args, input_text)
    model = model.to(device)
    
    checkpoint_path = "outputs/RAER-FrameLevelFusion-[06-07]-[03:06]/model_best.pth"
    print(f"=> Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    from dataloader.video_dataloader import test_data_loader
    from torch.utils.data import DataLoader
    test_dataset = test_data_loader(
        root_dir="",
        list_file="RAER/annotation/test.txt",
        num_segments=16,
        duration=1,
        image_size=224,
        bounding_box_face="RAER/bounding_box/face.json",
        bounding_box_body="RAER/bounding_box/body.json",
        crop_body=False,
        num_classes=5
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    conditions = ["Original", "Face Only", "Body Only", "Context Only", "Face + Body"]
    results = {}
    
    for condition in conditions:
        war, uar = evaluate_ablation(model, test_loader, device, class_names, condition)
        results[condition] = {'WAR': war, 'UAR': uar}
        
    print("\n" + "="*50)
    print("ABLATION STUDY RESULTS (RAER)")
    print("="*50)
    print(f"{'Condition':<15} | {'WAR (%)':<10} | {'UAR (%)':<10}")
    print("-" * 50)
    for cond, metrics in results.items():
        print(f"{cond:<15} | {metrics['WAR']:<10.2f} | {metrics['UAR']:<10.2f}")
    print("="*50)

if __name__ == '__main__':
    main()
