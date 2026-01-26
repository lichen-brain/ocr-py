import os
import shutil
source_folder = "image_repo"
destination_folder = "image"
i = 1
for image_path in os.listdir(source_folder):
    if image_path.endswith(".png") or image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        shutil.copy(os.path.join(source_folder, image_path), destination_folder)
        new_name = "image_" + str(i) + "." + image_path.rsplit('.', 1)[1]
        new_file_path = os.path.join(destination_folder, new_name)
        os.rename(os.path.join(destination_folder, image_path), new_file_path)
        print(f"Copied and renamed {image_path} to {new_name}")
        i += 1
print("共复制并重命名了", i - 1, "张图片")