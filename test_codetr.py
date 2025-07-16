# import torch
# from mmcv import Config
# from Co_DETR.mmdet.apis import init_detector, inference_detector
#
# def infer_one_image(config_path, ckpt_path, image_path, score_thr=0.5, device='cuda'):
#     cfg = Config.fromfile(config_path)
#     model = init_detector(cfg, ckpt_path, device=device)
#     result = inference_detector(model, image_path)
#     out_file = 'result.jpg'
#     model.show_result(image_path, result, score_thr=score_thr, out_file=out_file)
#     print(f"Saved visualization to {out_file}")
#
# if __name__ == '__main__':
#     config_path = r"C:\Users\tam\Documents\GitHub\AIC\Co_DETR\projects\configs\co_dino\co_dino_5scale_swin_large_16e_o365tococo.py"
#     ckpt_path   = r"C:\Users\tam\Documents\GitHub\AIC\co_dino_5scale_swin_large_16e_o365tococo.pth"
#     image_path  = r"anh-cho-meo.jpg"
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     infer_one_image(config_path, ckpt_path, image_path, score_thr=0.5, device=device)

import torch
from mmcv import Config
from Co_DETR.mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

def infer_one_image(config_path, ckpt_path, image_path, score_thr=0.5, device='cuda'):
    """
    Object and Color Detection:
    Use Co-DETR + Swin-L to detect objects, extract dominant color inside each bbox,
    and draw thicker bbox with color names.
    """
    # Load config and model
    cfg = Config.fromfile(config_path)
    model = init_detector(cfg, ckpt_path, device=device)

    # Run detection
    result = inference_detector(model, image_path)
    img = cv2.imread(image_path)

    # Define basic colors and KDTree
    basic_colors = {
        'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
        'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255),
        'black': (0, 0, 0), 'white': (255, 255, 255), 'gray': (128, 128, 128),
        'orange': (255, 165, 0), 'brown': (165, 42, 42), 'pink': (255, 192, 203),
        'purple': (128, 0, 128)
    }
    color_names = list(basic_colors.keys())
    color_values = np.array(list(basic_colors.values()))
    tree = KDTree(color_values)

    # Co-DETR trả về list kết quả, mỗi phần tử là array shape (N, 5): x1, y1, x2, y2, score
    if isinstance(result, tuple):
        result = result[0]  # bỏ segmentation mask nếu có

    for class_id, class_result in enumerate(result):
        for bbox in class_result:
            x1, y1, x2, y2, score = bbox
            class_name = model.CLASSES[class_id]
            if score < score_thr:
                continue
            # Crop object từ ảnh gốc
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Lấy dominant color trong object
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).reshape(-1, 3)
            kmeans = KMeans(n_clusters=1, random_state=42).fit(crop_rgb)
            dom_color = kmeans.cluster_centers_[0].astype(int)
            dist, idx = tree.query([dom_color], k=1)
            color_name = color_names[idx[0][0]]

            # Vẽ bbox dày hơn (thickness=3) + màu bbox gần với dominant color
            bbox_color = tuple(int(c) for c in dom_color[::-1])
            cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=3)

            # Vẽ viền đen trước (thickness=4)
            cv2.putText(img, f"{color_name} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
            # Sau đó vẽ text màu lên trên (thickness=2)
            cv2.putText(img, f"{color_name} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, thickness=2, lineType=cv2.LINE_AA)

            # In thông tin ra console
            print(f"{class_name} | score={score:.2f} | dominant color={dom_color.tolist()} | nearest color={color_name} | bbox=({x1},{y1},{x2},{y2})")

    out_file = 'result_with_colors.jpg'
    cv2.imwrite(out_file, img)
    print(f"Saved result with colored bboxes to {out_file}")

if __name__ == '__main__':
    config_path = r"C:\Users\tam\Documents\GitHub\AIC\Co_DETR\projects\configs\co_dino\co_dino_5scale_swin_large_16e_o365tococo.py"
    ckpt_path = r"C:\Users\tam\Documents\GitHub\AIC\co_dino_5scale_swin_large_16e_o365tococo.pth"
    image_path = r"anh-cho-meo.jpg"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    infer_one_image(config_path, ckpt_path, image_path, score_thr=0.5, device=device)
