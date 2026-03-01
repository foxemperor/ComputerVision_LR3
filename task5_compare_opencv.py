"""
ЛАБОРАТОРНАЯ РАБОТА №3 - ЗАДАНИЕ 5
Сравнение ручной реализации со встроенным cv2.GaussianBlur()
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import create_gaussian_kernel, normalize_kernel, apply_convolution, calculate_psnr


def main():
    print("ЗАДАНИЕ 5: Сравнение с cv2.GaussianBlur()")
    print("=" * 60)

    image_path = "images/input.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Поместите тестовое изображение в {image_path}")
        return

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(f"Исходное изображение: {original.shape}")

    size = 5
    sigma = 1.5
    print(f"\n Параметры: ядро {size}×{size}, σ={sigma}")

    # --- Ручная реализация ---
    kernel = create_gaussian_kernel(size, sigma)
    kernel = normalize_kernel(kernel)
    manual_result = apply_convolution(original, kernel)
    psnr_manual = calculate_psnr(original, manual_result)
    print(f"   [Ручная] PSNR = {psnr_manual:.2f} dB")

    # --- OpenCV встроенный метод ---
    opencv_result = cv2.GaussianBlur(original, (size, size), sigmaX=sigma, sigmaY=sigma)
    psnr_opencv = calculate_psnr(original, opencv_result)
    print(f"   [OpenCV] PSNR = {psnr_opencv:.2f} dB")

    # --- Разница между реализациями ---
    diff = cv2.absdiff(manual_result, opencv_result)
    max_diff = int(np.max(diff))
    mean_diff = float(np.mean(diff))
    print(f"\n Разница (ручная vs OpenCV):")
    print(f"   Макс. отклонение = {max_diff} px")
    print(f"   Среднее отклонение = {mean_diff:.4f} px")

    # --- Сохранение результатов ---
    cv2.imwrite("output/task5_manual.png", manual_result)
    cv2.imwrite("output/task5_opencv.png", opencv_result)
    diff_vis = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("output/task5_diff.png", diff_vis)

    # --- Сводный график ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Оригинал", fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(manual_result, cmap='gray')
    axes[1].set_title(f"Ручная реализация\nPSNR={psnr_manual:.1f} dB", fontsize=10)
    axes[1].axis('off')

    axes[2].imshow(opencv_result, cmap='gray')
    axes[2].set_title(f"cv2.GaussianBlur()\nPSNR={psnr_opencv:.1f} dB", fontsize=10)
    axes[2].axis('off')

    axes[3].imshow(diff_vis, cmap='hot')
    axes[3].set_title(f"Разница (усилена)\nmax={max_diff} px, mean={mean_diff:.2f}", fontsize=10)
    axes[3].axis('off')

    fig.suptitle("Задание 5: Ручная реализация vs cv2.GaussianBlur()", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/task5_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("\n✅ Результаты сохранены:")
    print("   - output/task5_manual.png")
    print("   - output/task5_opencv.png")
    print("   - output/task5_diff.png")
    print("   - output/task5_comparison.png")


if __name__ == "__main__":
    main()
