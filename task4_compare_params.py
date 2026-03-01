"""
ЛАБОРАТОРНАЯ РАБОТА №3 - ЗАДАНИЕ 4
Сравнение фильтра Гаусса при разных σ и размерах ядра
на ОДНОМ изображении
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import create_gaussian_kernel, normalize_kernel, apply_convolution, calculate_psnr


def main():
    print("ЗАДАНИЕ 4: Сравнение параметров фильтра Гаусса")
    print("=" * 60)

    image_path = "images/input.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Поместите тестовое изображение в {image_path}")
        return

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(f" Исходное изображение: {original.shape}")

    # 2 разных σ и 2 разных размера ядра → 4 комбинации
    params = [
        {"size": 3, "sigma": 1.0},
        {"size": 3, "sigma": 2.5},
        {"size": 7, "sigma": 1.0},
        {"size": 7, "sigma": 2.5},
    ]

    results = []
    for p in params:
        size, sigma = p["size"], p["sigma"]
        print(f"\n Ядро {size}×{size}, σ={sigma}")

        kernel = create_gaussian_kernel(size, sigma)
        kernel = normalize_kernel(kernel)
        filtered = apply_convolution(original, kernel)

        psnr = calculate_psnr(original, filtered)
        print(f"PSNR = {psnr:.2f} dB")

        results.append({
            "label": f"{size}×{size}, σ={sigma}",
            "image": filtered,
            "psnr": psnr,
        })

        out_name = f"output/task4_size{size}_sigma{sigma}.png"
        cv2.imwrite(out_name, filtered)
        print(f"Сохранено: {out_name}")

    # Сводный график: оригинал + 4 результата
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Оригинал", fontsize=11)
    axes[0].axis('off')

    for i, res in enumerate(results):
        axes[i + 1].imshow(res["image"], cmap='gray')
        axes[i + 1].set_title(f"{res['label']}\nPSNR={res['psnr']:.1f} dB", fontsize=10)
        axes[i + 1].axis('off')

    fig.suptitle("Задание 4: Сравнение параметров Гауссова фильтра", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/task4_comparison_all.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ Сводное сравнение: output/task4_comparison_all.png")


if __name__ == "__main__":
    main()
