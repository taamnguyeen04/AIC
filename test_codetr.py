import cv2
import numpy as np
import sys
from typing import List, Tuple, Dict
from mmcv import Config
from Co_DETR.mmdet.apis import init_detector, inference_detector
from sklearn.cluster import KMeans
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from pprint import pprint



class ObjectColorDetector:
    """Sử dụng Co-DETR để phát hiện đối tượng và phân tích màu sắc chủ đạo của toàn ảnh."""
    def __init__(self, device: str = 'cuda:0'):
        self.device = device
        config_path = r"C:\Users\tam\Documents\GitHub\AIC\Co_DETR\projects\configs\co_dino\co_dino_5scale_swin_large_16e_o365tococo.py"
        ckpt_path = r"C:\Users\tam\Documents\GitHub\AIC\co_dino_5scale_swin_large_16e_o365tococo.pth"

        sys.path.append('/kaggle/working/Co_DETR')
        self.model = init_detector(Config.fromfile(config_path), ckpt_path, device=self.device)

        # --- Danh sách màu cơ bản (RGB) ---
        self.basic_colors = {
            'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
            'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255),
            'black': (0, 0, 0), 'white': (255, 255, 255), 'gray': (128, 128, 128),
            'orange': (255, 165, 0), 'brown': (165, 42, 42), 'pink': (255, 192, 203),
            'purple': (128, 0, 128)
        }
        self.basic_colors_lab = self._convert_basic_colors_to_lab()

    def _convert_basic_colors_to_lab(self) -> dict:
        """Chuyển basic_colors sang CIELAB để so sánh nhanh hơn."""
        lab_dict = {}
        for name, rgb in self.basic_colors.items():
            rgb_obj = sRGBColor(*rgb, is_upscaled=True)
            lab_obj = convert_color(rgb_obj, LabColor)
            lab_dict[name] = lab_obj
        return lab_dict

    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB tuple to CIELAB tuple."""
        rgb_obj = sRGBColor(*rgb, is_upscaled=True)
        lab_obj = convert_color(rgb_obj, LabColor)
        return (lab_obj.lab_l, lab_obj.lab_a, lab_obj.lab_b)

    def _get_closest_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Tìm tên màu gần nhất theo thị giác (CIELAB + Delta E CIEDE2000)."""
        rgb_color = sRGBColor(*rgb, is_upscaled=True)
        lab_color = convert_color(rgb_color, LabColor)

        min_delta = float('inf')
        closest_name = None

        for name, lab_ref in self.basic_colors_lab.items():
            delta = delta_e_cie2000(lab_color, lab_ref)
            if delta < min_delta:
                min_delta = delta
                closest_name = name

        return closest_name

    def detect(self, image_path: str) -> Tuple[List[Tuple[float, float, float]], Dict[str, List[Tuple[float, float, float, Tuple[int, int, int, int]]]]]:
        try:
            result = inferen_detector(self.model, image_path)
            if isinstance(result, tuple):
                result = result[0]

            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Không đọc được ảnh.")

            # --- MÀU CHỦ ĐẠO TOÀN ẢNH ---
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            flat_pixels = img_rgb.reshape(-1, 3)
            kmeans = KMeans(n_clusters=6, random_state=42, n_init=10).fit(flat_pixels)
            dominant_rgb = kmeans.cluster_centers_.astype(int)

            dominant_colors_lab = [self._rgb_to_lab(tuple(color)) for color in dominant_rgb]

            # --- MÀU CỦA TỪNG OBJECT ---
            object_colors_lab = {}

            for class_id, bboxes in enumerate(result):
                if class_id >= len(self.model.CLASSES):
                    continue
                class_name = self.model.CLASSES[class_id]
                for bbox in bboxes:
                    if bbox[4] < 0.5:
                        continue
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).reshape(-1, 3)
                    kmeans_obj = KMeans(n_clusters=1, random_state=0, n_init=10).fit(crop_rgb)
                    dom_rgb = kmeans_obj.cluster_centers_[0].astype(int)
                    lab_color = self._rgb_to_lab(tuple(dom_rgb))

                    if class_name not in object_colors_lab:
                        object_colors_lab[class_name] = []
                    # Thêm vị trí bounding box vào kết quả
                    object_colors_lab[class_name].append((lab_color, (x1, y1, x2, y2)))

            return dominant_colors_lab, object_colors_lab

        except Exception as e:
            print(f"Error during detection: {e}")
            return [], {}

def main():
    config_path = r"C:\Users\tam\Documents\GitHub\AIC\Co_DETR\projects\configs\co_dino\co_dino_5scale_swin_large_16e_o365tococo.py"
    ckpt_path = r"C:\Users\tam\Documents\GitHub\AIC\co_dino_5scale_swin_large_16e_o365tococo.pth"
    image_path = r"C:\Users\tam\Documents\data\keyframes_output\L02_L02_V002_0034.72s.jpg"
    detector = ObjectColorDetector(device='cuda:0')

    dominant_colors_lab, object_colors_lab = detector.detect(image_path)

    print("color_filters")
    pprint(dominant_colors_lab)

    print("object_filters")
    pprint(object_colors_lab)

    for label in object_colors_lab.keys():
        for value in object_colors_lab[label]:
            color, bbox = value
            print(f"{label}: {color}, {bbox}")
            valuee = ",".join(map(str, list(color) + list(bbox)))  # Chuyển color và bbox thành list rồi nối
            print(valuee)


if __name__ == "__main__":
    main()


# if __name__ == '__main__':
#     config_path = r"C:\Users\tam\Documents\GitHub\AIC\Co_DETR\projects\configs\co_dino\co_dino_5scale_swin_large_16e_o365tococo.py"
#     ckpt_path = r"C:\Users\tam\Documents\GitHub\AIC\co_dino_5scale_swin_large_16e_o365tococo.pth"
#     image_path = r"C:\Users\tam\Documents\data\keyframes_output\L02_L02_V002_0034.72s.jpg"
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     infer_one_image(config_path, ckpt_path, image_path, score_thr=0.5, device=device)
