import argparse
import os
import torch
import cv2
import numpy as np
import time
import threading

from models.Generate_Model import GenerateModel
from models.clip import clip
from utils.builders import get_class_info

class Args:
    pass

def setup_args(checkpoint_path):
    args = Args()
    args.dataset = "RAER"
    args.text_type = "prompt_ensemble"
    args.class_names_with_context = "False"
    args.temporal_layers = 1
    args.contexts_number = 8
    args.class_token_position = "end"
    args.class_specific_contexts = "True"
    args.load_and_tune_prompt_learner = "True"
    args.clip_path = "ViT-B/16"
    args.drop_path_rate = 0.0
    args.image_size = 224
    args.crop_body = False
    args.use_moco = False
    args.temperature = 0.07
    
    if torch.backends.mps.is_available():
        args.device = torch.device("mps")
    elif torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    args.freeze_image_encoder = False
    return args

def load_model(args, checkpoint_path):
    print("Loading CLIP model...")
    clip_model, _ = clip.load(args.clip_path, device=args.device)
    clip_model.float() 
    class_names, input_text = get_class_info(args)
    print("Initializing RAPT-CLIP-RAER model...")
    model = GenerateModel(input_text, clip_model, args)
    model.to(args.device)
    model.float() 
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    model.eval()
    return model, class_names

def preprocess_image(img_bgr):
    """Convert BGR image to CLIP input tensor format [1, 1, C, H, W]."""
    img_resized = cv2.resize(img_bgr, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    img_norm = (img_norm - mean) / std
    img_chw = np.transpose(img_norm, (2, 0, 1))
    return torch.tensor(img_chw, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Shared variables
latest_results = []
is_running = True
frame_to_process = None
lock = threading.Lock()

def inference_thread(model, class_names, args):
    global latest_results, is_running, frame_to_process, lock
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Persistence: Keep results for a few frames if detection fails
    last_valid_results = []
    frames_since_last_detection = 0
    MAX_PERSISTENCE = 10 # frames

    while is_running:
        with lock:
            frame = frame_to_process.copy() if frame_to_process is not None else None
        if frame is None:
            time.sleep(0.01)
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Optimized detection parameters for stability
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))
        
        if len(faces) > 0:
            face_tensors = []
            valid_boxes = []
            
            for (x, y, w, h) in faces:
                y1, y2, x1, x2 = max(0, y), min(frame.shape[0], y+h), max(0, x), min(frame.shape[1], x+w)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0: continue
                face_tensors.append(preprocess_image(face_img))
                valid_boxes.append((x1, y1, x2-x1, y2-y1))
            
            if face_tensors:
                # BATCH INFERENCE for speed
                # face_batch: [N, 1, 3, 224, 224]
                face_batch = torch.cat(face_tensors, dim=0).to(args.device)
                body_tensor = preprocess_image(frame).to(args.device)
                # repeat body_tensor for each face in batch
                body_batch = body_tensor.repeat(len(face_tensors), 1, 1, 1, 1)
                
                try:
                    with torch.no_grad():
                        output, _, _, _ = model(face_batch, body_batch)
                        preds = torch.argmax(output, dim=-1).cpu().numpy()
                    
                    current_results = []
                    for idx, pred_idx in enumerate(preds):
                        current_results.append({
                            'pred_class': class_names[pred_idx],
                            'face_box': valid_boxes[idx]
                        })
                    
                    last_valid_results = current_results
                    frames_since_last_detection = 0
                    with lock:
                        latest_results = current_results
                except Exception as e:
                    print(f"Inference error: {e}")
        else:
            # Detection failed, use persistence
            frames_since_last_detection += 1
            if frames_since_last_detection < MAX_PERSISTENCE:
                with lock:
                    latest_results = last_valid_results
            else:
                with lock:
                    latest_results = []

def main():
    global latest_results, is_running, frame_to_process, lock
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--camera_id", type=int, default=0)
    args_cmd = parser.parse_args()
    
    args = setup_args(args_cmd.checkpoint)
    model, class_names = load_model(args, args_cmd.checkpoint)
    
    worker = threading.Thread(target=inference_thread, args=(model, class_names, args))
    worker.daemon = True
    worker.start()
    
    cap = cv2.VideoCapture(args_cmd.camera_id)
    if not cap.isOpened(): return
        
    print("Real-time Emotion Recognition started. Multi-person optimized. Press 'q' to quit.")
    
    fps_time = time.time()
    fps_frames = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        with lock:
            frame_to_process = frame
            results = latest_results
            
        display_frame = frame.copy()
        for res in results:
            pred_class = res['pred_class']
            x, y, w, h = res['face_box']
            
            # Simple Smoothing for display
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Label with background
            (text_w, text_h), _ = cv2.getTextSize(pred_class, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (x, y - text_h - 10), (x + text_w + 5, y), (0, 255, 0), -1)
            cv2.putText(display_frame, pred_class, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            
        fps_frames += 1
        if time.time() - fps_time >= 1.0:
            fps, fps_frames, fps_time = fps_frames, 0, time.time()
        cv2.putText(display_frame, f"FPS: {fps} | People: {len(results)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('RAPT-CLIP Emotion Recognition', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    is_running = False
    cap.release()
    cv2.destroyAllWindows()
    worker.join(timeout=1.0)

if __name__ == "__main__":
    main()
