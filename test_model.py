import torch
import argparse
from models.Generate_Model import GenerateModel
from models.clip import clip

args = argparse.Namespace(
    device='cpu',
    dataset='EMOTIC',
    backbone='ViT-B/16',
    use_context=True,
    fusion_type='cmaf',
    is_ensemble=False,
    use_cocoop=False,
    use_label_gcn=False,
    use_moco=True,
    moco_dim=512,
    moco_k=65536,
    moco_m=0.999,
    moco_t=0.07,
    contexts_number=4,
    class_token_position='end',
    class_specific_contexts='False',
    load_and_tune_prompt_learner='False',
    temporal_layers=1,
    temperature=0.07,
    lambda_mi=0.0,
)

input_text = ['A photo of someone feeling Happiness']
clip_model, _ = clip.load(args.backbone, device='cpu', jit=False)

print("Instantiating GenerateModel...")
model = GenerateModel(input_text, clip_model, args)
print("Model initialized successfully!")

B = 2
dummy_face = torch.randn(B, 1, 3, 224, 224)
dummy_body = torch.randn(B, 1, 3, 224, 224)
dummy_context = torch.randn(B, 1, 3, 224, 224)

print("Running forward pass...")
out, text_feats, hand_feats, moco_logits = model(dummy_face, dummy_body, dummy_context)
print("Forward output shape:", out.shape)
assert out.shape == (B, len(input_text)), f"Expected shape {(B, len(input_text))}, got {out.shape}"

print("Running momentum forward pass...")
out_m = model.forward_momentum(dummy_face, dummy_body, dummy_context)
print("Momentum output shape:", out_m.shape)
assert out_m.shape == (B, 512), f"Expected shape {(B, 512)}, got {out_m.shape}"

print("Success! Structural changes are sound.")
