import os
import csv
import random
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils import data
import torchvision

# =========================
# 1) Label mapping (DAiSEE)
# =========================
DAISEE_LABEL_MAP = {
    "engagement": 0,
    "boredom": 1,
    "confusion": 2,
    "frustration": 3
}

# =========================
# 2) Simple transforms (giống RAER style)
# Bạn có thể thay bằng GroupResize/Stack/ToTorchFormatTensor của bạn
# nếu muốn giữ y hệt pipeline.
# =========================
def default_train_transform(image_size=224):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomRotation(4),
        torchvision.transforms.ToTensor(),  # (C,H,W) float [0,1]
    ])

def default_test_transform(image_size=224):
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
    ])

# =========================
# 3) TSN sampling (giống RAER)
# =========================
def get_train_indices(num_frames, num_segments, duration):
    average_duration = (num_frames - duration + 1) // num_segments
    if average_duration > 0:
        offsets = np.multiply(list(range(num_segments)), average_duration) + \
                  np.random.randint(average_duration, size=num_segments)
    elif num_frames > num_segments:
        offsets = np.sort(np.random.randint(num_frames - duration + 1, size=num_segments))
    else:
        offsets = np.pad(np.array(list(range(num_frames))), (0, num_segments - num_frames), "edge")
    return offsets

def get_test_indices(num_frames, num_segments, duration):
    if num_frames > num_segments + duration - 1:
        tick = (num_frames - duration + 1) / float(num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
    else:
        offsets = np.pad(np.array(list(range(num_frames))), (0, num_segments - num_frames), "edge")
    return offsets

# =========================
# 4) (Optional) Face detector
# Cách nhẹ nhất: dùng OpenCV HaarCascade (nhanh, nhưng không “xịn”)
# Nếu muốn tốt hơn: MTCNN/facenet-pytorch (mình sẽ ghi hướng dẫn phía dưới)
# =========================
class HaarFaceCropper:
    def __init__(self):
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(haar_path)

    def crop_face(self, bgr_img, margin=20):
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return bgr_img  # fallback: trả ảnh gốc

        # lấy face lớn nhất
        x, y, w, h = sorted(faces, key=lambda t: t[2]*t[3], reverse=True)[0]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(bgr_img.shape[1], x + w + margin)
        y2 = min(bgr_img.shape[0], y + h + margin)
        return bgr_img[y1:y2, x1:x2]

# =========================
# 5) DAiSEE Dataset (đọc video mp4)
# =========================
class DaiseeVideoDataset(data.Dataset):
    """
    CSV format: video_path,label
      - video_path: đường dẫn tương đối hoặc tuyệt đối đến .mp4
      - label: 0..3 hoặc string: Engagement/Boredom/Confusion/Frustration
    """
    def __init__(
        self,
        csv_file,
        root_dir="",
        mode="train",
        num_segments=16,
        duration=1,
        image_size=224,
        transform=None,
        use_face=True,
        face_cropper=None
    ):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.mode = mode
        self.num_segments = num_segments
        self.duration = duration
        self.image_size = image_size

        self.transform = transform if transform is not None else (
            default_train_transform(image_size) if mode == "train" else default_test_transform(image_size)
        )

        self.use_face = use_face
        self.face_cropper = face_cropper if face_cropper is not None else HaarFaceCropper()

        self.samples = self._read_csv(csv_file)
        print(f"[DAiSEE] Loaded {len(self.samples)} videos from {csv_file}")

    def _read_csv(self, csv_file):
        items = []
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                rel_path, label = row[0].strip(), row[1].strip()

                # map label nếu là string
                if not label.isdigit():
                    key = label.lower()
                    if key in DAISEE_LABEL_MAP:
                        label_id = DAISEE_LABEL_MAP[key]
                    else:
                        raise ValueError(f"Unknown label string: {label}")
                else:
                    label_id = int(label)

                video_path = os.path.join(self.root_dir, rel_path) if self.root_dir else rel_path
                items.append((video_path, label_id))
        return items

    def _get_num_frames(self, cap):
        # OpenCV sometimes fails CAP_PROP_FRAME_COUNT; vẫn dùng tạm
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(n, 0)

    def _read_frame_at(self, cap, frame_idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame  # BGR

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label = self.samples[index]
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            # fallback: trả zeros
            dummy = torch.zeros((self.num_segments * self.duration, 3, self.image_size, self.image_size))
            return dummy, dummy, label

        num_frames = self._get_num_frames(cap)
        if num_frames <= 0:
            cap.release()
            dummy = torch.zeros((self.num_segments * self.duration, 3, self.image_size, self.image_size))
            return dummy, dummy, label

        if self.mode == "train":
            indices = get_train_indices(num_frames, self.num_segments, self.duration)
        else:
            indices = get_test_indices(num_frames, self.num_segments, self.duration)

        indices = np.clip(indices, 0, num_frames - 1)

        full_imgs = []
        face_imgs = []

        for seg_ind in indices:
            p = int(seg_ind)
            for _ in range(self.duration):
                frame = self._read_frame_at(cap, p)
                if frame is None:
                    # nếu lỗi read thì dùng frame trước đó hoặc skip
                    frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

                # full frame -> PIL RGB
                full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                full_pil = Image.fromarray(full_rgb)

                if self.use_face:
                    face_bgr = self.face_cropper.crop_face(frame, margin=20)
                    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                else:
                    face_pil = full_pil

                full_imgs.append(full_pil)
                face_imgs.append(face_pil)

                p = min(p + 1, num_frames - 1)

        cap.release()

        # transform từng frame (list -> stack)
        full_t = torch.stack([self.transform(im) for im in full_imgs], dim=0)   # (T, C, H, W)
        face_t = torch.stack([self.transform(im) for im in face_imgs], dim=0)   # (T, C, H, W)

        return face_t, full_t, label


# =========================
# 6) Factory functions giống RAER
# =========================
def train_data_loader_daisee(root_dir, csv_file, num_segments, duration, image_size, use_face=True):
    return DaiseeVideoDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        mode="train",
        num_segments=num_segments,
        duration=duration,
        image_size=image_size,
        transform=default_train_transform(image_size),
        use_face=use_face
    )

def test_data_loader_daisee(root_dir, csv_file, num_segments, duration, image_size, use_face=True):
    return DaiseeVideoDataset(
        csv_file=csv_file,
        root_dir=root_dir,
        mode="test",
        num_segments=num_segments,
        duration=duration,
        image_size=image_size,
        transform=default_test_transform(image_size),
        use_face=use_face
    )