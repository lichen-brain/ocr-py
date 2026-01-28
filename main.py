"""
快递单号识别系统 - 主入口
用法: python main.py [image_path]
"""

import sys
from pathlib import Path
from express_ocr import ExpressOCR

def main():
    print("=" * 60)
    print("快递单号识别系统 v1.0")
    print("=" * 60)
    
    ocr = ExpressOCR()
    
    if len(sys.argv) > 1:
        # 处理指定图片
        image_path = Path(sys.argv[1])
        if image_path.exists():
            result = ocr.recognize(image_path)
            print(f"\n识别结果: {result}")
        else:
            print(f"文件不存在: {image_path}")
    else:
        # 处理image目录下所有图片
        image_dir = Path(__file__).parent / "image"
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        results = ocr.batch_recognize(image_dir, output_dir)
        
        print("\n" + "=" * 60)
        print(f"识别结果汇总: {ocr.cnt_true} / {ocr.cnt_all}")
        print(f"正确率: {ocr.cnt_true / ocr.cnt_all * 100:.2f}%")
        print("=" * 60)
        for r in results:
            status = "✓" if r['is_true'] else " "
            print(f"  {status} {Path(r['image_path']).name}: {r['tracking_numbers']}  ->  {r['barcode_number'] if not r['is_true'] else ''}")

if __name__ == "__main__":
    main()
