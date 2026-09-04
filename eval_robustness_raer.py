import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import recall_score

from models.Generate_Model import GenerateModel
from models.clip import clip
from utils.builders import get_class_info, build_model

def get_args():
    parser = argparse.ArgumentParser(description='RAER Robustness Evaluation')
    parser.add_argument('--dataset', default='RAER')
    parser.add_argument('--text-type', default='prompt_ensemble')
    parser.add_argument('--class-names-with-context', default='False')
    parser.add_argument('--temporal-layers', default=1, type=int)
    parser.add_argument('--contexts-number', default=8, type=int)
    parser.add_argument('--class-token-position', default='end')
    parser.add_argument('--class-specific-contexts', default='True')
    parser.add_argument('--load-and-tune-prompt-learner', default='True')
    parser.add_argument('--clip-path', default='ViT-B/16')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--use-context', action='store_true')
    parser.add_argument('--fusion-type', default='cmaf', choices=['concat', 'cmaf', 'hierarchical'])
    parser.add_argument('--is-ensemble', action='store_true')
    parser.add_argument('--use-cocoop', action='store_true')
    parser.add_argument('--use-label-gcn', action='store_true')
    parser.add_argument('--drop_path_rate', default=0.0, type=float)
    parser.add_argument('--use-weighted-sampler', action='store_true')
    return parser.parse_args()

def add_noise(tensor, std=0.1):
    noise = torch.randn_like(tensor) * std
    return torch.clamp(tensor + noise, 0, 1) # Assuming input is normalized roughly 0-1, or just let it pass

def eval_robustness(model, test_loader, device):
    model.eval()
    
    noise_levels = [0.0, 0.1, 0.3, 0.5]
    results = {std: {'preds': [], 'targets': []} for std in noise_levels}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Robustness"):
            if len(batch) == 4:
                img_f, img_b, img_c, target = batch
            else:
                continue
                
            img_f = img_f.to(device)
            img_b = img_b.to(device)
            img_c = img_c.to(device)
            target = target.to(device)
            
            for std in noise_levels:
                noisy_f = img_f + torch.randn_like(img_f) * std
                noisy_b = img_b + torch.randn_like(img_b) * std
                noisy_c = img_c + torch.randn_like(img_c) * std
                
                output_tuple = model(noisy_f, noisy_b, noisy_c)
                output = output_tuple[0]
                _, predicted = torch.max(output.data, 1)
                
                results[std]['preds'].extend(predicted.cpu().numpy())
                results[std]['targets'].extend(target.cpu().numpy())
                
    print("\n" + "="*50)
    print("ROBUSTNESS EVALUATION (Gaussian Noise)")
    print("="*50)
    print(f"{'Noise Std':<15} | {'WAR (%)':<10} | {'UAR (%)':<10}")
    print("-" * 50)
    
    for std in noise_levels:
        preds = np.array(results[std]['preds'])
        targets = np.array(results[std]['targets'])
        
        war = (preds == targets).sum() / len(targets) * 100
        uar = np.mean(recall_score(targets, preds, average=None)) * 100
        print(f"{std:<15.2f} | {war:<10.2f} | {uar:<10.2f}")
    print("="*50)

def main():
    args = get_args()
    args.use_context = True 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available(): device = torch.device("mps")
    args.device = device
    args.freeze_image_encoder = False
    args.lr_image_encoder = 0
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
    
    eval_robustness(model, test_loader, device)

if __name__ == '__main__':
    main()
