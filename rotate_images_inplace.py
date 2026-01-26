from pathlib import Path
import cv2
import numpy as np


def rotate_images_ccw_90_inplace(image_dir: str) -> None:
    image_dir = Path(image_dir)
    print("扫描目录:", image_dir.resolve())

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}

    for img_path in image_dir.iterdir():
        if not img_path.is_file():
            continue

        if img_path.suffix.lower() not in valid_exts:
            print("跳过(后缀):", img_path.name)
            continue

        data = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            print("读取失败:", img_path.name)
            continue

        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imencode(img_path.suffix, rotated)[1].tofile(img_path)

        print("已覆盖:", img_path.name)


if __name__ == "__main__":
    rotate_images_ccw_90_inplace("images")
