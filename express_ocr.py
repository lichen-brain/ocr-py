"""
快递单号识别模块 v1.0
"""

import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import re
from pathlib import Path
from pyzbar import pyzbar
import zxingcpp


class ExpressOCR:
    def __init__(self):
        self.tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract版本: {self.tesseract_version}")
        self.cnt_all = 0
        self.cnt_true = 0
        self.ocr_config_6 = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789JYTD'
        self.ocr_config_7 = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789JYTD'
        self.ocr_config_8 = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789JYTD'
        self.TRACKING_REGEX = re.compile(
            r'((JD\d{13})|(SF\d{13})|(JT\d{13})|(YT\d{13})|(77\d{13})|(7\d{13})|([3|4]\d{14})|(\d{13,15}))'
        )
            # r'^(.*[JD|YT|JT]?.*)$'
    
    def detect_barcode_in_top(self, image):
        """在顶部 1/4 or 全图 区域检测条形码"""
        h, w = image.shape[:2]
        image = cv2.resize(image, (h // 5, w // 5))
        # top_region = image[ : h // 4, : ]  # 顶部1/4
        top_region = image  # 全图
        image_contour = top_region.copy()
        
        gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
        
        # Sobel梯度检测条形码特征
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(cv2.convertScaleAbs(gradX), cv2.convertScaleAbs(gradY))
        
        # 模糊和二值化
        blurred = cv2.blur(gradient, (9, 9))
        _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
        
        # 闭运算填充
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=10)
        closed = cv2.dilate(closed, None, iterations=10)
        
        # 查找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        barcodes = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = bw / float(bh) if bh > 0 else 0
            area = bw * bh
            
            # 条形码特征：宽高比>3，面积适中
            if aspect_ratio > 2.5 and area > 3000 and bw > 80:
                barcodes.append((x, y, bw, bh))
        
        # 按面积排序，取最大的
        barcodes.sort(key=lambda b: b[2] * b[3], reverse=True)

        y = barcodes[0][1] + barcodes[0][3] if barcodes else 0
        y_ = y + barcodes[0][3] * 3 // 5 if barcodes else image.shape[0]
        # y_ = y + barcodes[0][3] * 4 // 5 if barcodes else image.shape[0]
        # x = barcodes[0][0] + barcodes[0][2] // 5 if barcodes else 0
        # x_ = x + barcodes[0][2] - barcodes[0][2] // 5 if barcodes else image.shape[1]
        x = barcodes[0][0] if barcodes else 0
        x_ = x + barcodes[0][2] if barcodes else image.shape[1]
        number_region = top_region[y : y_, x : x_] if y > 0 else top_region

        # # cv2.imshow("gray", gray)
        # # cv2.imshow("blurred", blurred)
        # # cv2.imshow("thresh", thresh)
        # cv2.imshow("closed", closed)
        # cv2.drawContours(image_contour, contours, -1, (0, 0, 255), 1)
        # cv2.imshow("Contour Image", image_contour)
        # # cv2.imshow("image", image)
        # cv2.imshow("number Region", number_region)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return number_region
    
    
    def ocr_image_region(self, image):
        image = cv2.resize(image, None, fx=3, fy=3)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2, 2)
        # blur = cv2.GaussianBlur(gray, (3, 3), 8, 8)
        adapt = cv2.adaptiveThreshold(gray, 255,
                    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 4)
        # adapt = cv2.adaptiveThreshold(blur, 255, 
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        
        variants = [
            image,
            gray,
            blur,
            # adapt,
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
            cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        ]

        # cv2.imshow("image", image)
        # cv2.imshow("gray", gray)
        # cv2.imshow("blur", blur)
        # cv2.imshow("adapt", adapt)
        # cv2.imshow("1", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        # cv2.imshow("2", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])
        # cv2.imshow("3", cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        candidates = []
        A_candidates = []

        for img in variants:
            psm = [pytesseract.image_to_string(img, config=self.ocr_config_6),
                   pytesseract.image_to_string(img, config=self.ocr_config_7),
                   pytesseract.image_to_string(img, config=self.ocr_config_8)
                ]
            for text in psm:
                # if re.match(r'^(JT|YT)?\d{10,15}$', text, re.IGNORECASE):
                #     candidates.append(text)
                text = re.sub(r'\s*', '', text).upper()
                text = self.TRACKING_REGEX.search(text)
                if text:
                    text = text.group()
                    if text.startswith(('YT', 'JT', 'JD')):
                        A_candidates.append(text)
                    candidates.append(text)

        if not candidates:
            return None

        from collections import Counter
        if A_candidates:
            return Counter(A_candidates).most_common(1)[0][0]
        return Counter(candidates).most_common(1)[0][0]
        # return candidates

    def extract_main_tracking_number_bar(self, image):
        # 策略1: 条形码解码（最准确）
        try:
            zx_results = zxingcpp.read_barcodes(image)
            for r in zx_results:
                data = r.text
                if re.match(r'^(JT|YT|JD)?\d{10,16}$', data, re.IGNORECASE):
                    print(f"    条形码解码: {data}")
                    return data.upper(), None
        except:
            pass
        
        # 策略2: pyzbar解码（备选）
        try:
            decoded = pyzbar.decode(image)
            for bc in decoded:
                data = bc.data.decode('utf-8')
                if re.match(r'^(JT|YT|JD)?\d{10,16}$', data, re.IGNORECASE):
                    print(f"    pyzbar解码: {data}")
                    return data.upper(), None
        except:
            pass
        
        print("  ✗ 未能通过条形码解码获取单号")

        return None, None

    def extract_main_tracking_number(self, image):
        """提取主单号 - OCR识别，条形码解码判断是否正确"""

        # 顶部条形码区域OCR
        cut = self.detect_barcode_in_top(image)
        number_ocr = self.ocr_image_region(cut)
        # print(f"  顶部条形码区域: {number_ocr}")

        # 条形码解码
        number_bar, _ = self.extract_main_tracking_number_bar(image)

        return number_ocr, number_bar
    
    def recognize(self, image_path, output_path=None):
        """识别单张图片"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取: {image_path}")
            return None
        
        print(f"\n处理: {Path(image_path).name} ({image.shape[1]}x{image.shape[0]})")
        
        # 1. 方向检测与矫正
        result_image = image.copy()
        h, w = image.shape[:2]
        
        tracking_numbers = []
        
        # 2. 使用新方法提取主单号
        number_ocr, number_bar = self.extract_main_tracking_number(image)
        if number_bar and number_ocr:
            if number_ocr == number_bar:
                self.cnt_true += 1
                print(f"  ✓ 识别到ocr单号: {number_ocr}")
            else:
                print(f"  ✗ ocr单号与条形码单号不符: {number_ocr} vs {number_bar}")
                print(f"ocr_len: {len(number_ocr)}   bar_len: {len(number_bar)}")
        elif not number_ocr:
            print(f"  ✗ 未识别到ocr单号")
        self.cnt_all += 1
        tracking_numbers.append(number_ocr)
        cv2.putText(result_image, f" {number_ocr}", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        
        # 保存结果
        if output_path is None:
            output_path = Path(image_path).parent / f"result_{Path(image_path).name}"
        cv2.imwrite(str(output_path), result_image)
        
        # print(f"  结果: {tracking_numbers if tracking_numbers else '未识别到'}")
        
        return {
            'is_true': number_ocr == number_bar,
            'image_path': str(image_path),
            'tracking_numbers': tracking_numbers,
            'barcode_number': number_bar,
            'output_path': str(output_path)
        }
    
    def batch_recognize(self, image_dir, output_dir):
        """批量识别"""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in image_dir.glob(ext):
                output_path = output_dir / f"result_{img_path.name}"
                result = self.recognize(img_path, output_path)
                if result:
                    results.append(result)        
        return results


# 兼容旧代码
def recognize_express_number(image_path, output_path=None):
    ocr = ExpressOCR()
    return ocr.recognize(image_path, output_path)


if __name__ == "__main__":
    ocr = ExpressOCR()
    image_dir = Path(__file__).parent / "image"
    output_dir = Path(__file__).parent / "output"
    ocr.batch_recognize(image_dir, output_dir)
