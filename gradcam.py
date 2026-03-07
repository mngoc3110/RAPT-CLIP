import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        if isinstance(output, tuple):
             self.activations = output[0].detach()
        else:
             self.activations = output.detach()
             
    def save_gradient(self, module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()
            
    def __call__(self, input_face, input_body, target_class=None):
        self.model.zero_grad()
        output, _, _, _ = self.model(input_face, input_body)
        
        # Ensure output is float32 for backward on MPS
        output = output.float()
        
        if target_class is None:
            target_class = torch.argmax(output, dim=-1)
            target_idx = target_class.item()
        else:
            if isinstance(target_class, torch.Tensor):
                target_idx = target_class.item()
            else:
                target_idx = target_class
                target_class = torch.tensor(target_idx).to(output.device)
            
        score = output[0, target_idx]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            print("Warning: Gradients or activations are None.")
            return np.zeros((14, 14), dtype=np.float32), target_idx

        # ViT typically outputs (L, B*T, D) where L = seq_len, B*T = batch * num_segments
        # CLIP's ViT uses (L, B*T, D), so we permute to (B*T, L, D) for easier indexing.
        acts = self.activations
        grads = self.gradients
        
        if acts.dim() == 3:
            # Permute from (L, B*T, D) to (B*T, L, D)
            acts = acts.permute(1, 0, 2)
            grads = grads.permute(1, 0, 2)
            
            # Pick the middle frame for visualization (most representative)
            mid_idx = min(acts.shape[0] // 2, acts.shape[0] - 1)
            activations = acts[mid_idx].cpu().float().numpy()
            gradients = grads[mid_idx].cpu().float().numpy()
        else:
            # Remove batch dim: [L, D]
            activations = acts[0].cpu().float().numpy()
            gradients = grads[0].cpu().float().numpy()
        
        # For ViT: Sequence length = 1 (CLS) + N (Spatial tokens)
        # We only care about spatial tokens (index 1 to end)
        spatial_activations = activations[1:] # [N, D]
        spatial_gradients = gradients[1:] # [N, D]
        
        # 1. Calculate weights: Global Average Pooling of gradients over spatial dimensions
        # weight shape: [D]
        weights = np.mean(spatial_gradients, axis=0)
        
        # 2. Linear combination of activations and weights
        # cam shape: [N]
        cam = np.zeros(spatial_activations.shape[0], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * spatial_activations[:, i]
            
        # 3. ReLU to keep only features that have a positive influence
        cam = np.maximum(cam, 0)
        
        # 4. Reshape to 2D grid
        grid_size = int(np.sqrt(cam.shape[0]))
        if grid_size * grid_size == cam.shape[0]:
            cam = cam.reshape(grid_size, grid_size)
        else:
            # Fallback if shape is weird
            cam = cam.reshape(-1, 1)
            
        # 5. Normalize between 0 and 1
        cam_min = np.min(cam)
        cam_max = np.max(cam)
        
        if cam_max - cam_min > 1e-7:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_idx

def overlay_cam_on_image(img, mask, alpha=0.5):
    # Ensure mask is scaled correctly before applying colormap
    mask_scaled = np.uint8(255 * mask)
    
    # Apply JET colormap (Red is hot, Blue is cold)
    heatmap = cv2.applyColorMap(mask_scaled, cv2.COLORMAP_JET)
    
    # Threshold the mask so we don't tint the whole image blue
    _, thresholded = cv2.threshold(mask_scaled, 50, 255, cv2.THRESH_BINARY)
    thresholded_3ch = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
    
    # If the image is float [0, 1], convert back to uint8 [0, 255] for OpenCV functions
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = np.uint8(255 * img)
        else:
            img = np.uint8(img)
            
    # Blend only where the threshold is active
    # This prevents the "blue screen" effect on areas with 0 activation
    overlayed = np.where(thresholded_3ch > 0, cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0), img)
    
    return overlayed

