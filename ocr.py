import cv2
import numpy as np
import matplotlib.pyplot as plt

# 图像预处理函数
def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法加载图像！")
        return None

    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 图像去噪（高斯模糊）
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 图像二值化（Otsu方法）
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image, image

# 提取水平投影特征
def horizontal_projection(binary_image):
    # 计算水平投影（每一行的黑白像素点的数量）
    projection = np.sum(binary_image, axis=1)
    return projection

# 提取垂直投影特征
def vertical_projection(binary_image):
    # 计算垂直投影（每一列的黑白像素点的数量）
    projection = np.sum(binary_image, axis=0)
    return projection

# 检测轮廓
def detect_contours(binary_image):
    # 找到所有轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 绘制轮廓
def draw_contours(image, contours):
    result_image = image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    return result_image

# 显示图像
def show_image(image, title="Image"):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 主函数
def main(image_path):
    # 预处理图像
    binary_image, original_image = preprocess_image(image_path)

    if binary_image is None:
        return

    # 显示原始图像和预处理后的二值图像
    show_image(binary_image, title="Binary Image")

    # 提取水平和垂直投影特征
    h_projection = horizontal_projection(binary_image)
    v_projection = vertical_projection(binary_image)

    # 显示投影特征
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(h_projection)
    plt.title("Horizontal Projection")
    plt.subplot(1, 2, 2)
    plt.plot(v_projection)
    plt.title("Vertical Projection")
    plt.show()

    # 检测并绘制轮廓
    contours = detect_contours(binary_image)
    contour_image = draw_contours(original_image, contours)

    # 显示带轮廓的图像
    show_image(contour_image, title="Contours Image")

if __name__ == "__main__":
    image_path = 'your_image_path_here.jpg'  # 替换为你的图像路径
    main(image_path)
