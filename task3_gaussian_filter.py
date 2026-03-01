"""
ЛАБОРАТОРНАЯ РАБОТА №3 - ЗАДАНИЕ 3
Реализация фильтра Гаусса средствами Python
"""

import cv2
import numpy as np
import os
from utils import (create_gaussian_kernel, normalize_kernel, apply_convolution,
                  print_kernel, compare_images, calculate_psnr)

def main():
    print("ЗАДАНИЕ 3: Ручная реализация фильтра Гаусса")
    print("=" * 60)
    
    # Загружаем тестовое изображение в grayscale
    image_path = "images/input.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Поместите тестовое изображение в {image_path}")
        return
    
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        print("❌ Ошибка загрузки изображения!")
        return
    
    print(f"Исходное изображение: {original.shape}")
    
    # Параметры фильтра
    size = 5
    sigma = 1.5
    
    print(f"\n Применяем фильтр: размер={size}×{size}, σ={sigma}")
    
    # 1. Создаём и нормализуем ядро
    kernel = create_gaussian_kernel(size, sigma)
    kernel = normalize_kernel(kernel)
    print_kernel(kernel, "Используемое ядро")
    
    # 2. Применяем свёртку (ручная реализация!)
    filtered_manual = apply_convolution(original, kernel)
    
    # 3. Сохраняем результат
    cv2.imwrite("output/task3_manual_result.png", filtered_manual)
    
    # 4. Вычисляем PSNR (для оценки качества)
    psnr = calculate_psnr(original, filtered_manual)
    print(f"📊 PSNR (качество): {psnr:.2f} dB")
    
    # 5. Создаём сравнение
    comparison = compare_images(original, filtered_manual, 
                               f"Manual Gaussian {size}×{size}, σ={sigma}")
    cv2.imwrite("output/task3_comparison.png", comparison)
    
    print("\n✅ Результаты сохранены:")
    print("   - output/task3_manual_result.png")
    print("   - output/task3_comparison.png")
    
    # 6. Показываем результаты
    cv2.imshow("Original", original)
    cv2.imshow("Manual Gaussian Filter", filtered_manual)
    cv2.imshow("Comparison", comparison)
    print("Нажмите любую клавишу для закрытия окон...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
