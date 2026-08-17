import torch
import numpy as np
from models.clip import clip
from models.Generate_Model import GenerateModel
from models.Text import prompt_ensemble_emotic
from dataloader.video_dataloader import VideoDataset
import torchvision.transforms as transforms
from dataloader.video_transform import Stack, ToTorchFormatTensor, GroupResize
from utils.loss import AsymmetricLoss
from sklearn.metrics import average_precision_score
import argparse

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
clip_model, _ = clip.load('ViT-B/16', device=device)
args = argparse.Namespace(
    device=device,
    dataset='EMOTIC', use_context=True, fusion_type='cmaf', is_multilabel=True,
    num_segments=1, temporal_layers=1, contexts_number=16, class_token_position='end',
    class_specific_contexts='True', load_and_tune_prompt_learner='True',
    temperature=0.07, use_moco=False, drop_path_rate=0.0, freeze_image_encoder=False,
    ablation_no_text=False, use_v2=False, modality_dropout=0.1, duration=1,
    image_size=224, lambda_mi=0.0, lambda_cad=0.0, lambda_text=0.0, use_q2l=False
)
model = GenerateModel(prompt_ensemble_emotic, clip_model, args)
model.to(device)

emotic_train_freq = torch.tensor([
    0.0558, 0.0110, 0.0217, 0.2692, 0.0106, 0.2230, 0.0196, 0.0555,
    0.0272, 0.0287, 0.0080, 0.4994, 0.0465, 0.2534, 0.0314, 0.0089,
    0.2572, 0.0079, 0.0777, 0.1048, 0.0196, 0.0208, 0.0135, 0.0199,
    0.0474, 0.0358
])
model.class_bias.data.zero_()

transform = transforms.Compose([GroupResize(224), Stack(), ToTorchFormatTensor()])
ds = VideoDataset('emotic_dataset/val_bbox.txt', 1, 1, 'test', transform, 224,
    'emotic_dataset/emotic_face_bboxes_mtcnn.json', 'emotic_dataset/emotic_body_bboxes.json',
    crop_body=True, mask_context_body=False, root_dir='emotic_dataset/cvpr_emotic')
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)

optimizer = torch.optim.AdamW([
    {'params': model.project_fc.parameters(), 'lr': 3e-5},
    {'params': model.cmaf.parameters(), 'lr': 3e-5},
    {'params': model.face_adapter.parameters(), 'lr': 3e-5},
    {'params': model.prompt_learner.parameters(), 'lr': 1e-5},
    {'params': [model.class_bias], 'lr': 1e-3},
], weight_decay=0.01)

def eval_mAP():
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for i, (face, body, ctx, tgt) in enumerate(loader):
            if i >= 10: break
            face, body, ctx, tgt = face.to(device), body.to(device), ctx.to(device), tgt.to(device)
            logits, _, _, _ = model(face, body, ctx)
            preds.append(torch.sigmoid(logits))
            targets.append(tgt)
    return average_precision_score(torch.cat(targets).cpu().numpy(), torch.cat(preds).cpu().numpy(), average='macro') * 100

print(f'Zero-shot mAP: {eval_mAP():.2f}%')

model.train()
for i, (face, body, ctx, tgt) in enumerate(loader):
    if i >= 60: break
    face, body, ctx, tgt = face.to(device), body.to(device), ctx.to(device), tgt.to(device)
    optimizer.zero_grad()
    logits, _, _, _ = model(face, body, ctx)
    loss = criterion(logits, tgt.float())
    loss.backward()
    
    cb_grad = model.class_bias.grad.norm().item() if model.class_bias.grad is not None else 0
    pl_grad = next(model.prompt_learner.parameters()).grad.norm().item() if next(model.prompt_learner.parameters()).grad is not None else 0
    fc_grad = next(model.project_fc.parameters()).grad.norm().item() if next(model.project_fc.parameters()).grad is not None else 0
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if i % 10 == 0:
        print(f'Step {i}: Loss = {loss.item():.4f} | Grads: CB={cb_grad:.4f}, PL={pl_grad:.4f}, FC={fc_grad:.4f}')

print(f'mAP after 60 steps: {eval_mAP():.2f}%')
