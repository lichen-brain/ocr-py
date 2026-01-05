import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore, Back, Style
import pytesseract



init()  # 初始化 colorama  
image_directory = 'image'  # 图像读取目录
ocr_config = '--psm 6'  # OCR配置参数


# 预处理图像
def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if(image is None):
        print(Fore.RED + "Error: Unable to load image ({})".format(image_path))
        return None
    print(Fore.GREEN + "Image loaded successfully ({})".format(image_path))
    image = cv2.resize(image, (800, 600))
    # 转换为灰度图像
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ret_simple, thresh_simple = cv2.threshold(image_gray, 140, 255, cv2.THRESH_BINARY)

    
    # thresh_adapt = cv2.adaptiveThreshold(image_gray, 255, 
    #     cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blocksize, 2)
    # cv2.imshow("Adaptive Threshold", thresh_adapt)

    # Otsu二值化
    # ret_otsu, image_otsu = cv2.threshold(image_gray, 0, 255,
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # # cv2.imshow("Otsu Image", image_otsu)
    # image_preprocessed = image_otsu

    # 高斯滤波进行去噪 + 自适应二值化
    image_blur = cv2.GaussianBlur(image_gray, (3, 3), 0, 0)
    # ret_otsu, image_otsu = cv2.threshold(image_blur, 0, 255,
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image_adapt = cv2.adaptiveThreshold(image_gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
    # cv2.imshow("Blurring + Otsu's Threshold", image_otsu)
    image_preprocessed = image_adapt

    # # Otsu二值化 + 高斯滤波去噪
    # ret_otsu, image_otsu = cv2.threshold(image_gray, 0, 255,
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # image_blur = cv2.GaussianBlur(image_otsu, (3, 3), 0, 0)
    # #cv2.imshow("Otsu's Threshold + Blurring", image_blur)
    # image_preprocessed = image_blur


    return image_preprocessed, image



# 设置阈值，根据水平投影裁剪文本行区域(水平投影) 
# return -> [[],[]...] 每个文本行的范围行的列表(二维)  else return None
# 大列表里装的是从第几行到第几行的二维列表
def horizontal_projection(binary_image, h_min_ratio = 0.3):# 最小比例阈值
    h_projection = np.sum(binary_image == 0, axis=1)   # 计算水平投影（每一行的黑像素点的数量）
    h_max_val = np.max(h_projection)   # 水平投影最大值
    h_threshold = h_max_val * h_min_ratio  # 水平投影阈值
    rows = np.where(h_projection >= h_threshold)[0]   # 满足阈值的行索引
    # 没有显著的水平投影
    if(len(rows) == 0):
        print(Fore.RED + "No significant horizontal projection found.")
        return None
    h_regions = binary_image[rows[0]:rows[-1], :]   # 裁剪出的文本行区域, 列不变
    # print(h_regions)
    # print(h_max_val)
    # for i in rows:
    #     print(Fore.BLUE + f"Row {i}: Horizontal projection value = {h_projection[i]}")
    return h_regions



# 设置阈值，使用垂直投影分割字符(垂直投影) 
# return -> [[[]],[[]]...] 每个字符的范围列的列表(三维)  else return []
# 大列表里装的是每个字符区域的二维列表[起始列，结束列]
def vertical_projection(binary_image, v_min_width = 5, v_min_ratio = 0.3):# 最小宽度阈值, 最小比例阈值
    v_projection = np.sum(binary_image == 0, axis=0)   # 计算垂直投影（每一列的白像素点的数量）
    v_max_val = np.max(v_projection)   # 垂直投影最大值
    v_threshold = v_max_val * v_min_ratio  # 垂直投影阈值
    cols = np.where(v_projection >= v_threshold)[0]   # 满足阈值的列索引
    # 没有显著的垂直投影
    if(len(cols) == 0):
        print(Fore.RED + "No significant vertical projection found.")
        return []
    char_regions = []   # 存储字符区域的列表
    v_start = cols[0]   # 字符区域起始列索引

    for i in range(1, len(cols)):
        # print(Fore.CYAN + f"Column {cols[i]}: Vertical projection value = {v_projection[cols[i]]}")
        if cols[i] != cols[i - 1] + 1:
            v_end = cols[i - 1]
            if v_end - v_start > v_min_width:
                char_regions.append([v_start, v_end])
            v_start = cols[i]
    # 添加最后一个字符区域
    if cols[-1] - v_start > v_min_width:
        char_regions.append([v_start, cols[-1]])
    v_regions = [binary_image[:, x1:x2] for x1, x2 in char_regions]  # 裁剪出的字符区域, 每个小区域行不变
    return v_regions



# -> 已废弃, 使用horizontal_projection和vertical_projection函数
# 水平和垂直投影特征提取     
# def projection_character(binary_image):
#     # 绘制水平和垂直投影图
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(h_projection)
#     plt.title("Horizontal Projection")

#     plt.subplot(1, 2, 2)
#     plt.plot(v_projection)
#     plt.title("Vertical Projection")
#     plt.suptitle("projection_character")
#     plt.show()



# main
def main():
    for image_path in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image_path)

        image_preprocess, image = preprocess_image(image_path)
        if(image_preprocess is None):
            return None
        
        contours, _ = cv2.findContours(image_preprocess, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_contour = image.copy()
        cv2.drawContours(image_contour, contours, -1, (0, 0, 255), 1)

        # 预处理图片
        cv2.imshow("Preprocessed Image", image_preprocess)
        cv2.imshow("Contour Image", image_contour)
        # cv2.imshow("Original Image", image)

        print(Fore.YELLOW + "Recognized Text:")
        print(text) 
        
        # 按 'q' 键退出
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  
            break
        # projection_character(image_preprocess)

        
        # 使用Tesseract进行OCR识别
        text = pytesseract.image_to_string(image_preprocess, lang='chi_sim')
   
   
    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()

