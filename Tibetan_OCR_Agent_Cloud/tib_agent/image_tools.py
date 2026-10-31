from pathlib import Path
import cv2
import numpy as np


def read_image(path: str):
    """兼容中文文件路径，失败时抛异常。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'无法读取图像：{Path(path).name}')
    return img


def analyze_image(path: str, sharpness_threshold: float = 65.0,
                  contrast_threshold: float = 35.0) -> dict:
    img = read_image(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    flags = []
    if sharpness < sharpness_threshold:
        flags.append('疑似模糊')
    if contrast < contrast_threshold:
        flags.append('低对比度')
    if img.shape[1] < 350:
        flags.append('宽度较低')
    return {
        'width': int(img.shape[1]), 'height': int(img.shape[0]),
        'sharpness_laplacian_var': round(sharpness, 3),
        'contrast_std': round(contrast, 3),
        'flags': flags,
        'needs_enhancement': bool(flags),
        'note': '图像质量阈值为启发式初值，需用真实验证集调优；不代表OCR准确率。',
    }


def enhance_image(input_path: str, output_path: str) -> str:
    """保守的CLAHE增强；可能破坏某些藏文字形，必须与原图对照。"""
    img = read_image(input_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    suffix = Path(output_path).suffix.lower() or '.png'
    ok, enc = cv2.imencode(suffix, enhanced)
    if not ok:
        raise RuntimeError('增强图像编码失败')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    enc.tofile(str(output_path))
    return output_path
