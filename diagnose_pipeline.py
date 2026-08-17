"""
Script chẩn đoán toàn bộ data pipeline cho EMOTIC.
Chạy local để kiểm tra xem bbox có khớp không, ảnh có đọc được không, label có đúng không.
"""
import json
import os
import sys

# Dùng local emotic_dataset directory
FACE_JSON = '/Users/macbook/Desktop/coding/projects/RAPT-CLIP-RAER/emotic_dataset/emotic_face_bboxes_mtcnn.json'
BODY_JSON = '/Users/macbook/Desktop/coding/projects/RAPT-CLIP-RAER/emotic_dataset/emotic_body_bboxes.json'
TRAIN_TXT = '/Users/macbook/Desktop/coding/projects/RAPT-CLIP-RAER/emotic_dataset/train.txt'
ROOT_DIR  = '/Users/macbook/Desktop/coding/projects/RAPT-CLIP-RAER/emotic_dataset/cvpr_emotic'

print("=" * 60)
print("1. Kiểm tra cấu trúc JSON bbox")
print("=" * 60)

with open(FACE_JSON, 'r') as f:
    face_json = json.load(f)
with open(BODY_JSON, 'r') as f:
    body_json = json.load(f)

face_keys = list(face_json.keys())[:5]
print(f"Tổng số video keys trong face_json: {len(face_json)}")
print(f"5 keys đầu tiên: {face_keys}")
print()

# Kiểm tra cấu trúc một entry
sample_key = face_keys[0]
sample_val = face_json[sample_key]
print(f"Cấu trúc của key '{sample_key}':")
if isinstance(sample_val, dict):
    sub_keys = list(sample_val.keys())[:5]
    print(f"  Sub-keys (frame keys): {sub_keys}")
    first_frame_key = sub_keys[0]
    print(f"  Giá trị của frame key '{first_frame_key}': {sample_val[first_frame_key]}")
else:
    print(f"  Giá trị trực tiếp: {sample_val}")

print()
print("=" * 60)
print("2. Kiểm tra train.txt - format và bounding box lookup")
print("=" * 60)

with open(TRAIN_TXT, 'r') as f:
    lines = f.readlines()

print(f"Tổng số dòng: {len(lines)}")
print(f"5 dòng đầu:")
for line in lines[:5]:
    print(f"  '{line.strip()}'")

print()
# Parse một dòng
line = lines[0].strip()
parts = line.split(' ')
if ',' in line:
    label_idx = next(i for i, p in enumerate(parts) if ',' in p)
    path = ' '.join(parts[:label_idx])
    label = parts[label_idx]
else:
    path = ' '.join(parts[:-2])
    label = parts[-1]

print(f"Path parsed: '{path}'")
print(f"Label parsed: '{label}'")

# Thử lookup trong face_json
full_path = os.path.join(ROOT_DIR, path)
print(f"\nFull path sẽ là: '{full_path}'")
print(f"File tồn tại không? {os.path.exists(full_path)}")

# Thử lookup key
rel_path = path
video_key = os.path.splitext(rel_path)[0]
print(f"\nKey tìm kiếm trong face_json: '{video_key}'")
print(f"Khớp exact match? {video_key in face_json}")

# Thử suffix match
if video_key not in face_json:
    parts_v = video_key.split('/')
    found = False
    for i in range(1, len(parts_v)):
        sub_key = '/'.join(parts_v[i:])
        if sub_key in face_json:
            print(f"Khớp suffix match với key: '{sub_key}'")
            print(f"Frame keys trong key này: {list(face_json[sub_key].keys())[:5]}")
            found = True
            break
    if not found:
        print("KHÔNG khớp bất kỳ key nào trong face_json!")
        print(f"Thử tìm partial match...")
        for k in face_keys:
            if path.split('/')[-1] in k or k.split('/')[-1] in path:
                print(f"  Possible match: '{k}'")

print()
print("=" * 60)
print("3. Thống kê tỷ lệ hit/miss bbox cho 100 samples đầu")
print("=" * 60)

hit_face = 0
miss_face = 0
hit_body = 0
miss_body = 0

for line in lines[:100]:
    line = line.strip()
    if not line:
        continue
    parts = line.split(' ')
    if ',' in line:
        label_idx = next(i for i, p in enumerate(parts) if ',' in p)
        path = ' '.join(parts[:label_idx])
    else:
        path = ' '.join(parts[:-2]) if len(parts) > 2 else parts[0]
    
    video_key = os.path.splitext(path)[0]
    
    # Lookup face
    matched = None
    if video_key in face_json:
        matched = video_key
    else:
        parts_v = video_key.split('/')
        for i in range(1, len(parts_v)):
            sub_key = '/'.join(parts_v[i:])
            if sub_key in face_json:
                matched = sub_key
                break
    
    if matched:
        hit_face += 1
    else:
        miss_face += 1
    
    # Lookup body
    matched_b = None
    if video_key in body_json:
        matched_b = video_key
    else:
        parts_v = video_key.split('/')
        for i in range(1, len(parts_v)):
            sub_key = '/'.join(parts_v[i:])
            if sub_key in body_json:
                matched_b = sub_key
                break
    
    if matched_b:
        hit_body += 1
    else:
        miss_body += 1

print(f"Face bbox: {hit_face}/100 hit ({hit_face}%), {miss_face} miss")
print(f"Body bbox: {hit_body}/100 hit ({hit_body}%), {miss_body} miss")
print()

if miss_face > 80:
    print("⚠️  CẢNH BÁO: Hơn 80% sample KHÔNG TÌM THẤY face bbox!")
    print("   → Face stream sẽ nhận full image thay vì cropped face")
    print("   → Model bị confuse giữa Face stream và Context stream")
elif miss_face > 30:
    print("⚠️  CẢNH BÁO: Hơn 30% sample bị miss face bbox")
else:
    print("✅ Face bbox lookup OK")

if miss_body > 80:
    print("⚠️  CẢNH BÁO: Hơn 80% sample KHÔNG TÌM THẤY body bbox!")
    print("   → Body crop và Context masking sẽ bị vô hiệu hóa")
elif miss_body > 30:
    print("⚠️  CẢNH BÁO: Hơn 30% sample bị miss body bbox")
else:
    print("✅ Body bbox lookup OK")
