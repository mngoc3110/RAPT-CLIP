# builders.py

import argparse
from typing import Tuple
import os
import torch
import torch.utils.data
from models.clip import clip

from dataloader.video_dataloader import train_data_loader, test_data_loader
from models.Generate_Model import GenerateModel
from models.Text import *
from utils.utils import *


def build_model(args: argparse.Namespace, input_text: list) -> torch.nn.Module:
    print("Loading pretrained CLIP model...")
    # Pass drop_path_rate to clip.load so it can be used during model construction
    CLIP_model, _ = clip.load(args.clip_path, device='cpu', drop_path=args.drop_path_rate)

    print("\nInput Text Prompts:")
    # Handle the case where input_text is a list of lists for prompt ensembling
    if any(isinstance(i, list) for i in input_text):
        for class_prompts in input_text:
            print(f"- Class: {class_prompts}")
    else:
        for text in input_text:
            print(text)


    print("\nInstantiating GenerateModel...")
    if hasattr(args, 'use_v2') and args.use_v2:
        from models.Generate_Model_v2 import GenerateModel_v2
        model = GenerateModel_v2(input_text=input_text, clip_model=CLIP_model, args=args)
    elif hasattr(args, 'ablation_no_text') and args.ablation_no_text:
        from models.Generate_Model_NoText import GenerateModel_NoText
        model = GenerateModel_NoText(clip_model=CLIP_model, args=args)
    else:
        model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)

    for name, param in model.named_parameters():
        param.requires_grad = False

    # Freeze CLIP image encoder if lr_image_encoder is 0
    # Otherwise, make it trainable.
    if args.lr_image_encoder > 0 and not args.freeze_image_encoder:
        for name, param in model.named_parameters():
            if "image_encoder" in name:
                param.requires_grad = True

    trainable_params_keywords = ["prompt_learner", "project_fc", "face_adapter", "classifier", "cross_attn", "cmaf", "gate_fc", "modality_importance", "adaptive_gate", "gate_proj",
                                  "ml_head"]       # learned multi-label classifier (per-class thresholds, EMOTIC only)
    if getattr(args, 'temporal_layers', 1) > 0 and getattr(args, 'num_segments', 1) > 1:
        trainable_params_keywords += ["temporal_net", "temporal_net_body"]
    # q2l_head is managed by Generate_Model.__init__ based on use_q2l flag — do NOT force here
    if getattr(args, 'use_q2l', False):
        trainable_params_keywords += ["q2l_head", "label_query", "label_decoder"]
    
    print('\nTrainable parameters:')
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in trainable_params_keywords):
            param.requires_grad = True
            print(f"- {name}")
    print('************************\n')

    return model


def get_class_info(args: argparse.Namespace) -> Tuple[list, list]:
    """
    根据数据集和文本类型获取 class_names 和 input_text（用于生成 CLIP 模型文本输入）。

    Returns:
        class_names: 类别名称，用于混淆矩阵等
        input_text: 输入文本，用于传入模型
    """
    dataset_name = args.dataset.strip()
    print(f"DEBUG: get_class_info processing dataset '{dataset_name}'")

    if dataset_name == "RAER":
        class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']
        class_names_with_context = class_names_with_context_5
        class_descriptor = class_descriptor_5
        ensemble_prompts = prompt_ensemble_5
    elif dataset_name == "CK+":
        class_names = class_names_ckplus
        class_names_with_context = class_names_with_context_ckplus
        class_descriptor = class_descriptor_ckplus
        ensemble_prompts = prompt_ensemble_ckplus
    elif dataset_name == "SFER":
        class_names = class_names_sfer
        class_names_with_context = class_names_with_context_sfer
        class_descriptor = class_descriptor_sfer
        ensemble_prompts = prompt_ensemble_sfer
    elif dataset_name == "DAiSEE":
        class_names = class_names_daisee
        class_names_with_context = class_names_with_context_daisee
        class_descriptor = class_descriptor_daisee
        ensemble_prompts = prompt_ensemble_daisee
    elif dataset_name == "CAER":
        class_names = class_names_caer
        class_names_with_context = class_names_with_context_caer
        class_descriptor = class_descriptor_caer
        ensemble_prompts = prompt_ensemble_caer
    elif dataset_name == "EMOTIC":
        from models.Text import class_names_emotic, class_names_with_context_emotic, class_descriptor_emotic, prompt_ensemble_emotic
        class_names = class_names_emotic
        class_names_with_context = class_names_with_context_emotic
        class_descriptor = class_descriptor_emotic
        ensemble_prompts = prompt_ensemble_emotic
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented yet.")

    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor
    elif args.text_type == "prompt_ensemble":
        input_text = ensemble_prompts
    elif args.text_type == "au_guided_prompts":
        if dataset_name == "RAER":
            from models.Text import class_descriptor_5_au
            input_text = class_descriptor_5_au
        else:
            raise ValueError("AU-guided prompts are only defined for RAER dataset currently.")
    else:
        raise ValueError(f"Unknown text_type: {args.text_type}")

    return class_names, input_text



def build_dataloaders(args: argparse.Namespace) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]: 
    train_annotation_file_path = args.train_annotation
    val_annotation_file_path = args.val_annotation
    test_annotation_file_path = args.test_annotation
    
    class_names, _ = get_class_info(args)
    num_classes = len(class_names)

    # Debug print
    print(f"DEBUG: args.dataset = '{args.dataset}'")

    mask_context_body = getattr(args, 'mask_context_body', False)
    if mask_context_body:
        print("=> Background Masking Enabled: Masking body bounding box in context stream (CAER-Net style)")

    print(f"Loading train data (Standard) for {args.dataset}...")
    train_data = train_data_loader(
        root_dir=args.root_dir, list_file=train_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size, dataset_name=args.dataset,
        bounding_box_face=args.bounding_box_face, bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        mask_context_body=mask_context_body,
        num_classes=num_classes
    )
    
    print(f"Loading validation data (Standard) for {args.dataset}...")
    val_data = test_data_loader(
        root_dir=args.root_dir, list_file=val_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=args.bounding_box_face, bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        mask_context_body=mask_context_body,
        num_classes=num_classes
    )

    print(f"Loading test data (Standard) for {args.dataset}...")
    test_data = test_data_loader(
        root_dir=args.root_dir, list_file=test_annotation_file_path, num_segments=args.num_segments,
        duration=args.duration, image_size=args.image_size,
        bounding_box_face=args.bounding_box_face, bounding_box_body=args.bounding_box_body,
        crop_body=args.crop_body,
        mask_context_body=mask_context_body,
        num_classes=num_classes
    )

    print(f"Total number of training images: {len(train_data)}")
    print("Creating DataLoader instances...")
    
    sampler = None
    shuffle = True
    if args.use_weighted_sampler:
        print("=> Using WeightedRandomSampler.")
        is_multilabel_dataset = (args.dataset == "EMOTIC")
        if is_multilabel_dataset:
            # EMOTIC: multi-hot labels (e.g., "0,1,0,...,1 x1 y1 x2 y2")
            # Strategy: weight each sample by 1 / freq_of_rarest_positive_label.
            # This up-samples images that contain rare classes (Embarrassment, Fear, etc.),
            # giving the model more balanced exposure across all 26 categories.
            from dataloader.video_dataloader import VideoRecord
            all_records = train_data.video_list  # VideoRecord objects with .label tensor
            # Compute per-class frequencies
            all_labels_np = torch.stack([r.label for r in all_records]).numpy()  # (N, C)
            class_freq = all_labels_np.mean(axis=0)  # (C,) in [0,1]
            inv_freq = 1.0 / (class_freq + 1e-6)    # rare class → high weight
            # Per sample: weight = max over its positive classes of 1/freq
            sample_weights = []
            for rec in all_records:
                label_np = rec.label.numpy()  # (C,) binary
                pos_indices = label_np.nonzero()[0]
                if len(pos_indices) > 0:
                    w = float(inv_freq[pos_indices].max())
                else:
                    w = 1.0  # no positive label (shouldn't happen)
                sample_weights.append(w)
            sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        else:
            # Single-label datasets (RAER, CAER-S, DAiSEE, etc.)
            class_counts = get_class_counts(train_annotation_file_path)
            class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
            raw_labels = []
            with open(train_annotation_file_path, 'r') as f:
                for line in f:
                    raw_labels.append(int(line.strip().split()[-1]))
            min_label = min(raw_labels) if raw_labels else 0
            sample_weights = torch.tensor(
                [class_weights[label - min_label] for label in raw_labels], dtype=torch.float
            )
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False  # sampler and shuffle are mutually exclusive

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader