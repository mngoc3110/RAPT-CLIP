# run_inference_student.py
import argparse
import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

from models.Student_Model import StudentModel

# Define standard CLIP normalization transforms (re-used for Student)
IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
])

def load_student_model(checkpoint_path, num_classes=5, num_segments=16, device='cpu'):
    """Loads the distilled student model from checkpoint."""
    print(f"=> Loading Student Model from {checkpoint_path} onto {device}...")
    model = StudentModel(num_classes=num_classes, num_segments=num_segments)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    print("=> Student Model loaded successfully.")
    return model

def crop_face_and_body(image_bgr, face_cascade=None):
    """
    Extracts face and body crops from a raw frame.
    Uses Haar Cascades for face detection as fallback, or returns default crops.
    """
    h, w, c = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    
    face_box = None
    if face_cascade is not None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            # Sort by area size, pick largest face
            faces = sorted(faces, key=lambda bbox: bbox[2] * bbox[3], reverse=True)
            x, y, fw, fh = faces[0]
            face_box = (x, y, x + fw, y + fh)

    # 1. Face Crop (fallback to center crop if no face detected)
    if face_box is not None:
        face_img = img_pil.crop(face_box)
    else:
        # Fallback: crop center region of upper half
        face_img = img_pil.crop((w // 4, h // 8, 3 * w // 4, 5 * h // 8))

    # 2. Body Crop (entire frame, or face region blacked out)
    # To match 'crop-body' and RAPT-CLIP: we black out the face region in the frame
    body_img = img_pil.copy()
    if face_box is not None:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(body_img)
        draw.rectangle([face_box[0], face_box[1], face_box[2], face_box[3]], fill=(0, 0, 0))
    
    return face_img, body_img

def process_video(video_path, model, face_cascade, num_segments=16, device='cpu'):
    """
    Samples frames from a video, processes Face & Body streams, 
    and runs Student inference to predict emotion.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None
        
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames == 0:
        print("Error: Video has 0 frames.")
        cap.release()
        return None

    # Middle frame temporal sampling matching 'test' indices
    tick = num_frames / float(num_segments)
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
    offsets = np.clip(offsets, 0, num_frames - 1)

    face_tensors = []
    body_tensors = []

    print(f"Processing {num_segments} sampled segments from {video_path}...")
    start_time = time.time()
    
    for idx in offsets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Fallback to black frame
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
        face_pil, body_pil = crop_face_and_body(frame, face_cascade)
        
        face_tensors.append(transform(face_pil))
        body_tensors.append(transform(body_pil))

    cap.release()
    preprocess_time = (time.time() - start_time) * 1000

    # Stack into [1, T, 3, H, W] to match model inputs
    face_input = torch.stack(face_tensors, dim=0).unsqueeze(0).to(device)
    body_input = torch.stack(body_tensors, dim=0).unsqueeze(0).to(device)

    # Inference
    start_inference = time.time()
    with torch.no_grad():
        logits, _ = model(face_input, body_input)
        probs = F.softmax(logits, dim=-1).squeeze(0)
    inference_time = (time.time() - start_inference) * 1000

    return probs.cpu().numpy(), preprocess_time, inference_time

def main():
    parser = argparse.ArgumentParser(description="VPS-optimized Student Model Inference")
    parser.add_argument('--video', type=str, required=True, help="Path to input video file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to student_best.pth.")
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help="Hardware device.")
    args = parser.parse_args()

    # Load Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Warning: Could not load OpenCV Haar cascade XML. Falling back to default center crops.")
        face_cascade = None

    class_names = ['Neutrality', 'Enjoyment', 'Confusion', 'Fatigue', 'Distraction']

    # 1. Load Student Model
    model = load_student_model(args.checkpoint, num_classes=len(class_names), device=args.device)

    # 2. Run Inference
    probs, prep_time, inf_time = process_video(args.video, model, face_cascade, device=args.device)
    
    if probs is not None:
        predicted_idx = np.argmax(probs)
        print(f"\n========================================")
        print(f"INFERENCE RESULTS:")
        print(f"Input Video: {args.video}")
        print(f"Predicted Emotion: {class_names[predicted_idx]} ({probs[predicted_idx]*100:.2f}%)")
        print("\nProbability Distribution:")
        for name, prob in zip(class_names, probs):
            print(f" - {name:12}: {prob*100:6.2f}%")
        print(f"----------------------------------------")
        print(f"Timing Metrics:")
        print(f" - Bounding box crop & preprocess : {prep_time:.1f} ms")
        print(f" - Dual-stream model inference     : {inf_time:.1f} ms (Total {1000/inf_time:.1f} FPS)")
        print(f"========================================\n")

if __name__ == '__main__':
    main()
