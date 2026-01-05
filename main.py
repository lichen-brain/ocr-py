import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from colorama import init, Fore, Back, Style
import pytesseract



init()  # 初始化 colorama  
image_directory = '.\image'  # 图像读取目录


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
    ret_otsu, image_otsu = cv2.threshold(image_gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("Otsu Image", image_otsu)
    image_preprocessed = image_otsu

    # # 高斯滤波进行去噪 + Otsu二值化
    # image_blur = cv2.GaussianBlur(image_gray, (3, 3), 0, 0)
    # ret_otsu, image_otsu = cv2.threshold(image_blur, 0, 255,
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("Blurring + Otsu's Threshold", image_otsu)
    # image_preprocessed = image_otsu

    # # Otsu二值化 + 高斯滤波去噪
    # ret_otsu, image_otsu = cv2.threshold(image_gray, 0, 255,
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # image_blur = cv2.GaussianBlur(image_otsu, (3, 3), 0, 0)
    # cv2.imshow("Otsu's Threshold + Blurring", image_blur)
    # image_preprocessed = image_blur


    # key = cv2.waitKey(0)
    # if key == ord('q'):  # 按 'q' 键退出
    #     cv2.destroyAllWindows()
    #     sys.exit(0)
    return image_preprocessed, image

# main
def main():
    for image_path in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image_path)

        image_preprocess, image = preprocess_image(image_path)
        if(image_preprocess is None):
            return None

        # 预处理图片
        cv2.imshow("Preprocessed Image", image_preprocess)
        cv2.imshow("Original Image", image)
        # 使用Tesseract进行OCR识别
        text = pytesseract.image_to_string(image_preprocess, lang='chi_sim', config=r'--oem 3 --psm 6')

        print(Fore.YELLOW + "Recognized Text:")
        print(text) 

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # 按 'q' 键退出
            break
            
   
   
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()

