"""
快递单号识别模块 v2.0
优化：方向检测 + 顶部1/4条形码定位 + 精确单号识别
"""

import cv2
import pytesseract
import numpy as np
import re
from pathlib import Path
from pyzbar import pyzbar
import zxingcpp


class ExpressOCR:
    def __init__(self):
        self.tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract版本: {self.tesseract_version}")
        self.number_bar = ''
        self.cnt_all = 0
        self.cnt_true = 0
        self.ocr_config_row = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789JYT'

    def detect_orientation(self, image):
        """检测图片方向 - 仅在明确需要时旋转"""
        # 暂时禁用自动旋转，因为误判率较高
        # 快递单通常已经是正向的
        return image
    
    def detect_barcode_in_top(self, image):
        """在顶部 1/4 or 全图 区域检测条形码"""
        image = cv2.resize(image, (800, 600))
        # h, w = image.shape[:2]
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
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9))
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
        x = barcodes[0][0] + barcodes[0][2] // 5 if barcodes else 0
        x_ = x + barcodes[0][2] - barcodes[0][2] // 5 if barcodes else image.shape[1]
        # x = barcodes[0][0] if barcodes else 0
        # x_ = x + barcodes[0][2] if barcodes else image.shape[1]
        number_region = top_region[y : y_, x : x_] if y > 0 else top_region

        # # cv2.imshow("gray", gray)
        # # cv2.imshow("blurred", blurred)
        # # cv2.imshow("thresh", thresh)
        # cv2.imshow("closed", closed)
        cv2.drawContours(image_contour, contours, -1, (0, 0, 255), 1)
        cv2.imshow("Contour Image", image_contour)
        # # cv2.imshow("image", image)
        # cv2.imshow("number Region", number_region)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return number_region
    
    def ocr_image_region(self, image):
        """OCR识别图像区域"""
        image = cv2.resize(image, (image.shape[1] * 3, image.shape[0] * 3)) 
        # 灰度化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Black-hat, 去背景
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        # blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # # closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        # closed = cv2.erode(gray, None, iterations=1)
        # closed = cv2.dilate(closed, None, iterations=1)

        # 高斯滤波
        blur = cv2.GaussianBlur(gray, (3, 3), 8, 8)

        # # 自适应二值化
        # adapt = cv2.adaptiveThreshold(blur, 255, 
        #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

        # Otsu二值化
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        cv2.imshow("number Region", image)
        # cv2.imshow("erosion", closed)
        # cv2.imshow("blur", blur) 
        cv2.imshow("otsu", otsu)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        str = pytesseract.image_to_string(otsu , config=self.ocr_config_row)

        return str



    def extract_main_tracking_number_bar(self, image):
        # 策略1: 条形码解码（最准确）
        try:
            zx_results = zxingcpp.read_barcodes(image)
            for r in zx_results:
                data = r.text
                if re.match(r'^(JT)?\d{10,16}$', data, re.IGNORECASE):
                    print(f"    条形码解码: {data}")
                    return data.upper() if data.upper().startswith('JT') else data, None
        except:
            pass
        
        # 策略2: pyzbar解码（备选）
        try:
            decoded = pyzbar.decode(image)
            for bc in decoded:
                data = bc.data.decode('utf-8')
                if re.match(r'^(JT)?\d{10,16}$', data, re.IGNORECASE):
                    print(f"    pyzbar解码: {data}")
                    return data.upper() if data.upper().startswith('JT') else data, None
        except:
            pass
        
        print("  ✗ 未能通过条形码解码获取单号")

        return None, None

    def extract_main_tracking_number(self, image):
        """提取主单号 - 条形码解码优先，OCR备选"""
        h, w = image.shape[:2]
        
         
        x = self.detect_barcode_in_top(image)
        xx = self.ocr_image_region(x)
        print(f"  顶部条形码区域: {xx}")

        # 优先尝试条形码解码
        self.number_bar, _ = self.extract_main_tracking_number_bar(image)
        
        # 策略2: 多区域多预处理搜索
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        all_texts = []
        # 全图多种预处理
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        all_texts.append(pytesseract.image_to_string(otsu, config='--oem 3 --psm 6'))
        all_texts.append(pytesseract.image_to_string(gray, config='--oem 3 --psm 6'))
        
        # 顶部1/2区域（某些图片需要）
        top_half = gray[:h//2, :]
        all_texts.append(pytesseract.image_to_string(top_half, config='--oem 3 --psm 6'))
        
        text_clean = re.sub(r'\s+', '', '\n'.join(all_texts))
        
        # 极兔单号优先
        jt_nums = re.findall(r'JT\d{13,16}', text_clean, re.IGNORECASE)
        if jt_nums:
            from collections import Counter
            best = Counter(jt_nums).most_common(1)[0][0]
            print(f"    全图找到极兔单号: {best}")
            return best.upper(), None
        
        # 纯数字单号（13-16位）
        pure_nums = re.findall(r'\d{13,16}', text_clean)
        if pure_nums:
            from collections import Counter
            best = Counter(pure_nums).most_common(1)[0][0]
            print(f"    全图找到数字单号: {best}")
            return best, None
        
        # 扩大范围：12-16位
        pure_nums2 = re.findall(r'\d{12,16}', text_clean)
        if pure_nums2:
            from collections import Counter
            best = Counter(pure_nums2).most_common(1)[0][0]
            print(f"    全图找到数字单号(扩展): {best}")
            return best, None

        return None, None
    
    def recognize(self, image_path, output_path=None):
        """识别单张图片"""
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取: {image_path}")
            return None
        
        print(f"\n处理: {Path(image_path).name} ({image.shape[1]}x{image.shape[0]})")
        
        # 1. 方向检测与矫正
        image = self.detect_orientation(image)
        result_image = image.copy()
        h, w = image.shape[:2]
        
        tracking_numbers = []
        
        # 2. 使用新方法提取主单号
        number, _ = self.extract_main_tracking_number(image)
        if number:
            if self.number_bar and number != self.number_bar:
                print(f"  ✗ 主单号与条形码单号不符: {number} vs {self.number_bar}")
            else:
                self.cnt_true += 1
            tracking_numbers.append(number)
            cv2.putText(result_image, f" {number}", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
            print(f"  ✓ 识别到主单号: {number}")
        self.cnt_all += 1 
        
        # 保存结果
        if output_path is None:
            output_path = Path(image_path).parent / f"result_{Path(image_path).name}"
        cv2.imwrite(str(output_path), result_image)
        
        print(f"  结果: {tracking_numbers if tracking_numbers else '未识别到'}")
        
        return {
            'image_path': str(image_path),
            'tracking_numbers': tracking_numbers,
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
