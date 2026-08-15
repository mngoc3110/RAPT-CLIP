import os
import glob
import json
import random
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils import data
from numpy.random import randint
from dataloader.video_transform import *

# Custom Transform for List of Images (Group Transform)
class GroupRandomGrayscale(object):
    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            # Convert to Grayscale (L) then back to RGB to keep 3 channels
            return [img.convert('L').convert('RGB') for img in img_group]
        return img_group

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self): # 路径
        return self._data[0]

    @property       # 帧数
    def num_frames(self):
        if len(self._data) >= 3:
            return int(self._data[1])
        return 1

    @property       # 标签
    def label(self):
        # Support EMOTIC multi-label (comma separated string -> float tensor)
        raw_label = self._data[-1]
        if isinstance(raw_label, torch.Tensor):
            return raw_label
        if isinstance(raw_label, str) and ',' in raw_label:
            return torch.tensor([float(x) for x in raw_label.split(',')], dtype=torch.float32)
        return int(raw_label)

class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size,bounding_box_face,bounding_box_body, crop_body=False, root_dir="", num_classes=8):
        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.bounding_box_face = bounding_box_face
        self.bounding_box_body = bounding_box_body
        self.crop_body = crop_body
        self.root_dir = root_dir
        
        # Debugging: Initialize for saving sample images
        self.debug_samples_path = 'debug_samples'
        os.makedirs(self.debug_samples_path, exist_ok=True)
        self._saved_samples = {i: 0 for i in range(num_classes)}
        
        self._read_sample()
        self._parse_list()
        self._read_boxs()
        if self.crop_body: # Only read body boxes if cropping is enabled
            self._read_body_boxes()

    def _read_boxs(self):
        with open(self.bounding_box_face, 'r') as f:
            self.boxs = json.load(f)


    
    def _read_body_boxes(self):
        if self.bounding_box_body:
            with open(self.bounding_box_body, 'r') as f:
                self.body_boxes = json.load(f)


    def _cv2pil(self,im_cv):
        cv_img_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
        pillow_img = Image.fromarray(cv_img_rgb.astype('uint8'))
        return pillow_img

    def _pil2cv(self,im_pil):
        cv_img_rgb = np.array(im_pil)
        cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)
        return cv_img_bgr

    def _resize_image(self,im, width, height):
        w, h = im.shape[1], im.shape[0]
        r = min(width / w, height / h)
        new_w, new_h = int(w * r), int(h * r)
        im = cv2.resize(im, (new_w, new_h))
        pw = (width - new_w) // 2
        ph = (height - new_h) // 2
        top, bottom = ph, ph
        left, right = pw, pw
        if top + bottom + new_h < height:
            bottom += 1
        if left + right + new_w < width:
            right += 1
        im = cv2.copyMakeBorder(im, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return im, r

    def _face_detect(self,img,box,margin,mode = 'face'):
        if box is None:
            if mode == 'face':
                # FALLBACK: Return original image instead of black image if no face detected
                # This helps prevent information loss in Test set
                return img
            return img
        else:
            left, upper, right, lower = box
            left = int(left)
            upper = int(upper)
            right = int(right)
            lower = int(lower)
            
            # Swap coordinates if they are inverted
            if left > right:
                left, right = right, left
            if upper > lower:
                upper, lower = lower, upper
                
            # Fallback to full image if the bounding box is degenerate or extremely small
            if right - left < 2 or lower - upper < 2:
                return img
                
            left = max(0, left - margin)
            upper = max(0, upper - margin)
            right = min(img.width, right + margin)
            lower = min(img.height, lower + margin)
            
            # Final safety check before cropping
            if right <= left: right = left + 1
            if lower <= upper: lower = upper + 1
            if mode == 'face':
                img = img.crop((left, upper, right, lower))
                return img
            elif mode == 'body':
                occluded_image = img.copy()
                draw = ImageDraw.Draw(occluded_image)
                draw.rectangle([left, upper, right, lower], fill=(0, 0, 0))
                return occluded_image
    
    def _read_sample(self):
        # tmp = [x.strip().split(' ') for x in open(self.list_file)]
        # self.sample_list = [item for item in tmp]
        
        self.sample_list = []
        with open(self.list_file, 'r') as f:
            for line in f:
                if ',' in line:
                    parts = line.strip().split(' ')
                    # label is the first part that contains a comma
                    label_idx = next(i for i, part in enumerate(parts) if ',' in part)
                    path = ' '.join(parts[:label_idx])
                    label = parts[label_idx]
                    num_frames = 1
                    self.sample_list.append([path, num_frames, label])
                else:
                    parts = line.strip().split(' ')
                    if len(parts) > 3:
                        # Path contains spaces, join all parts except the last two
                        path = ' '.join(parts[:-2])
                        num_frames = parts[-2]
                        label = parts[-1]
                        self.sample_list.append([path, num_frames, label])
                    else:
                        self.sample_list.append(parts)


    def _parse_list(self):
        # 
        # Data Form: [video_id, num_frames, class_idx]
        # 
        self.video_list = []
        
        # Auto-detect label offset (0-based vs 1-based)
        min_label = 0
        try:
            scalar_labels = [int(item[-1]) for item in self.sample_list if not (isinstance(item[-1], str) and ',' in item[-1])]
            if len(scalar_labels) > 0:
                min_label = min(scalar_labels)
        except:
            pass
        self.label_offset = min_label # Subtract this offset to make it 0-based

        for item in self.sample_list:
            path = item[0]
            # Fix Kaggle dataset structure (Kaggle unzips directories to root)
            if path.startswith('CAER/'):
                path = path.replace('CAER/', '', 1)
            if 'kaggle' in self.root_dir.lower():
                path = path.replace('CAER_Video/', '')
                path = path.replace('dataset/RAER/', 'RAER/')
            
            if len(item) == 2:
                num_frames = 1
                label = item[1]
            elif len(item) >= 3:
                num_frames = int(item[-2])
                label = item[-1]
                
            self.video_list.append(VideoRecord([os.path.join(self.root_dir, path), num_frames, label]))
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in each part randomly
        # 
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _get_test_indices(self, record):
        # 
        # Split all frames into seg parts, then select frame in the mid of each part
        # 
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        # Check if record.path is a directory (frames) or a file (video)
        if os.path.isdir(record.path):
            video_frames_path = glob.glob(os.path.join(record.path, '*'))
            video_frames_path.sort()
            num_real_frames = len(video_frames_path)
            is_video_file = False
        elif record.path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            video_frames_path = [record.path]
            num_real_frames = 1
            is_video_file = False
        else:
            # Assume it's a video file
            is_video_file = True
            cap = cv2.VideoCapture(record.path)
            num_real_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Don't release cap yet, we need it to read frames
            if not cap.isOpened():
                 print(f"Warning: Could not open video file {record.path}, returning zeros.")
                 num_real_frames = 0

        if num_real_frames == 0:
            print(f"Warning: No frames found for video {record.path}, returning zeros.")
            dummy_shape = (self.num_segments * self.duration, 3, self.image_size, self.image_size)
            if is_video_file and 'cap' in locals(): cap.release()
            
            label_out = record.label if isinstance(record.label, torch.Tensor) else record.label - self.label_offset
            return torch.zeros(dummy_shape), torch.zeros(dummy_shape), torch.zeros(dummy_shape), label_out

        # Clamp indices to be valid
        indices = np.clip(indices, 0, num_real_frames - 1)
        
        images = list()
        images_face = list()
        images_context = list()
        
        current_frame_idx = -1
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                img_pil = None
                box = None
                
                # 1. Read Image — smart sequential seek to avoid expensive cap.set() on every frame
                if is_video_file:
                    gap = p - current_frame_idx
                    if gap == 1:
                        # Next consecutive frame: just read directly
                        ret, frame = cap.read()
                    elif 1 < gap <= 5:
                        # Small gap: use grab() to skip without decoding (much faster than seek)
                        for _ in range(gap - 1):
                            cap.grab()
                        ret, frame = cap.read()
                    else:
                        # Large gap or backward jump: forced seek
                        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                        ret, frame = cap.read()
                    current_frame_idx = p

                    if ret:
                        img_cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_cv_rgb)
                    else:
                        # Failed to read frame, use black image
                        img_pil = Image.new('RGB', (self.image_size, self.image_size))
                else:
                    img_path = video_frames_path[p]
                    try:
                        img_pil = Image.open(img_path).convert('RGB')
                    except:
                        img_pil = Image.new('RGB', (self.image_size, self.image_size))

                # 2. Key Lookup Strategy for Bounding Box
                # Construct possible keys to look up in the JSON
                # Priority 1: Full relative path from dataset root (e.g., 'RAER/train/Neutral/001')
                # Priority 2: Parent dir + Filename (e.g., 'Neutral/001')
                
                # Normalize path separators to forward slash
                rel_path = record.path.replace('./', '').replace('\\', '/')
                # Remove extension
                video_key_full = os.path.splitext(rel_path)[0]
                
                frame_key = f"{p}.jpg" # Standard frame key format
                
                # Try finding the video key in boxes
                matched_video_key = None
                
                # Strategy A: Exact match
                if video_key_full in self.boxs:
                    matched_video_key = video_key_full
                
                # Strategy B: Suffix match (handle 'dataset/' prefix issues)
                if matched_video_key is None:
                    parts = video_key_full.split('/')
                    for idx in range(1, len(parts)):
                        sub_key = '/'.join(parts[idx:])
                        if sub_key in self.boxs:
                            matched_video_key = sub_key
                            break
                
                # 3. Retrieve Box
                if matched_video_key and frame_key in self.boxs[matched_video_key]:
                    box = self.boxs[matched_video_key][frame_key]
                
                # Debug logging for missing boxes (only once per video to avoid spam)
                if box is None and i == 0 and p == indices[0]: 
                    # Only log if it's the first frame of the first segment
                    # print(f"[DEBUG] Missing Box: Video='{video_key_full}', Frame='{frame_key}'. MatchedKey='{matched_video_key}'")
                    pass

                # 4. Face Detection (Crop)
                # Reduce margin to 10 (Tight Crop) to zoom in on micro-expressions (eyebrows/eyes)
                # This helps separate Neutral vs Confusion
                # IMPORTANT FIX: Return full image if no box found (Fallback)
                img_pil_face = self._face_detect(img_pil, box, margin=10, mode='face')

                # 5. Body Crop (Optional)
                img_pil_body = img_pil # Default to full image
                if self.crop_body:
                    body_box = None
                    if matched_video_key and matched_video_key in self.body_boxes:
                        if frame_key in self.body_boxes[matched_video_key]:
                            body_box = self.body_boxes[matched_video_key][frame_key]
                    
                    if body_box is not None:
                        left, upper, right, lower = body_box
                        # Ensure coordinates are within image bounds
                        left = max(0, left); upper = max(0, upper)
                        right = min(img_pil.width, right); lower = min(img_pil.height, lower)
                        if right > left and lower > upper:
                            img_pil_body = img_pil.crop((left, upper, right, lower))

                # 6. Resize and Stack
                # Resize Body
                img_cv_body = self._pil2cv(img_pil_body)
                img_cv_body, _ = self._resize_image(img_cv_body, self.image_size, self.image_size)
                img_pil_body = self._cv2pil(img_cv_body)
                
                # Resize Face (face_detect returns cropped image, we need to resize it too?)
                # Wait, _face_detect returns cropped image but doesn't resize it to self.image_size?
                # The original code didn't resize face explicitly here, but transform does GroupResize.
                # However, for consistency, let's trust the transform pipeline which includes GroupResize.
                # But wait, `img_pil_face` might be tiny.
                
                images.append(img_pil_body)
                images_face.append(img_pil_face)
                images_context.append(img_pil)
                
                if p < num_real_frames - 1:
                    p += 1
        
        if is_video_file:
            cap.release()

        # Transforms take a list of PIL images
        # images_face: List[PIL.Image]
        # images: List[PIL.Image]
        
        # Apply transforms
        # Note: self.transform usually expects a list and returns a Tensor of shape (T*C, H, W)
        # ToTorchFormatTensor returns (C*T, H, W) because Stack concatenates on axis 2.
        
        process_data = self.transform(images) # (C*T, H, W)
        process_data_face = self.transform(images_face) # (C*T, H, W)
        process_data_context = self.transform(images_context) # (C*T, H, W)

        # Reshape to (T, C, H, W)
        # Since Stack concatenated [img1, img2, ...] -> [R1,G1,B1, R2,G2,B2, ...]
        # Reshaping to (-1, 3, H, W) correctly separates frames.
        process_data = process_data.view(-1, 3, self.image_size, self.image_size)
        process_data_face = process_data_face.view(-1, 3, self.image_size, self.image_size)
        process_data_context = process_data_context.view(-1, 3, self.image_size, self.image_size)
        
        label_out = record.label if isinstance(record.label, torch.Tensor) else record.label - self.label_offset
        return process_data_face, process_data, process_data_context, label_out

    def __len__(self):
        return len(self.video_list)


def train_data_loader(root_dir, list_file, num_segments, duration, image_size,dataset_name,bounding_box_face,bounding_box_body, crop_body=False, num_classes=8):
    if dataset_name in ["RAER", "CAER", "EMOTIC"]:
         train_transforms = torchvision.transforms.Compose([
            # Apply ColorJitter from video_transform (works on list of images)
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2), 
            GroupRandomGrayscale(p=0.2), # Custom transform for list of images
            RandomRotation(4),
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
    else:
         # Default transforms for other datasets like CK+
         train_transforms = torchvision.transforms.Compose([
            GroupResize(image_size),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchFormatTensor()])
            
    
    train_data = VideoDataset(root_dir=root_dir, list_file=list_file,
                              num_segments=num_segments, #16
                              duration=duration, #1
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size,
                              bounding_box_face=bounding_box_face,
                              bounding_box_body=bounding_box_body,
                              crop_body=crop_body,
                              num_classes=num_classes
                              )
    return train_data


def test_data_loader(root_dir, list_file, num_segments, duration, image_size,bounding_box_face,bounding_box_body, crop_body=False, num_classes=8):
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    
    test_data = VideoDataset(root_dir=root_dir, list_file=list_file,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size,
                             bounding_box_face=bounding_box_face,
                             bounding_box_body=bounding_box_body,
                             crop_body=crop_body,
                             num_classes=num_classes
                             )
    return test_data