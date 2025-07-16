import cv2
import os
import torch
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model từ OpenCLIP với pretrained dfn2b
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    model_name='ViT-L-14',
    pretrained='datacomp_xl_s13b_b90k',
    device=device
)

def extract_keyframes_with_timestamp(video_path, threshold=0.4, batch_size=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes = []
    prev_feature = None
    frame_batch = []
    index_batch = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
        frame_batch.append((frame, frame_pil))
        index_batch.append(frame_idx)

        if len(frame_batch) == batch_size:
            keyframes += process_batch_with_timestamp(frame_batch, index_batch, prev_feature, threshold, fps)
            prev_feature = keyframes[-1][2] if keyframes else prev_feature
            frame_batch, index_batch = [], []

        frame_idx += 1

    if frame_batch:
        keyframes += process_batch_with_timestamp(frame_batch, index_batch, prev_feature, threshold, fps)

    cap.release()
    return keyframes


def process_batch_with_timestamp(frame_batch, index_batch, prev_feature, threshold, fps):
    images_tensor = torch.cat([img_tensor for _, img_tensor in frame_batch], dim=0).to(device)

    with torch.no_grad():
        features = model.encode_image(images_tensor)
        features = features / features.norm(dim=-1, keepdim=True)  # normalize

    keyframes = []
    for i, (frame_data, feature, idx) in enumerate(zip(frame_batch, features, index_batch)):
        raw_frame, _ = frame_data
        feature = feature.unsqueeze(0)

        if prev_feature is None:
            keyframes.append((raw_frame, idx / fps, feature))
            prev_feature = feature.clone()
        else:
            diff = torch.norm(feature - prev_feature) / torch.norm(prev_feature)
            if diff.item() > threshold:
                keyframes.append((raw_frame, idx / fps, feature))
                prev_feature = feature.clone()

    return keyframes


def save_keyframes(keyframes, output_dir, folder_name, video_name):
    os.makedirs(output_dir, exist_ok=True)
    for idx, (frame, timestamp, _) in enumerate(keyframes):
        timestamp_str = f"{timestamp:07.2f}s"
        filename = f"{folder_name}_{video_name}_{timestamp_str}.jpg"
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, frame)


def process_all_videos(data_root, output_dir="keyframes_output", threshold=0.4):
    for folder in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder)
        for filename in os.listdir(folder_path):
            video_path = os.path.join(folder_path, filename)
            video_name = os.path.splitext(filename)[0]
            print(f"Xử lý: {video_path}")
            keyframes = extract_keyframes_with_timestamp(video_path, threshold=threshold)
            save_keyframes(keyframes, output_dir, folder, video_name)
            print(f"Đã lưu {len(keyframes)} keyframes từ {filename}")



if __name__ == "__main__":
    folder_with_videos = r"C:\Users\tam\Documents\data\AIC"
    process_all_videos(folder_with_videos)
