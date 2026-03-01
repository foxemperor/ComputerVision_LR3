"""
ЛАБОРАТОРНАЯ РАБОТА №3 - ЗАДАНИЕ 1
Построение матрицы ядра Гаусса
Размерности: 3x3, 5x5, 7x7
"""

import numpy as np
from utils import create_gaussian_kernel, print_kernel, save_kernel_to_file

def main():
    print("ЛАБОРАТОРНАЯ РАБОТА №3")
    print("ЗАДАНИЕ 1: Построение матрицы Гаусса")
    print("=" * 60)
    
    # Параметры для тестирования
    sizes = [3, 5, 7]
    sigma = 1.0  # Фиксированное σ для сравнения
    
    for size in sizes:
        print(f"\n Размер ядра: {size}×{size}")
        
        # Создаём ядро (пункты 1 и 2 алгоритма)
        kernel = create_gaussian_kernel(size, sigma)
        
        # Выводим в консоль
        print_kernel(kernel, f"Ядро Гаусса {size}×{size} (σ={sigma})")
        
        # Сохраняем в файл
        filename = f"output/task1_kernel_{size}x{size}_sigma{sigma}.txt"
        save_kernel_to_file(kernel, filename)
        
        print(f"Обработано размер {size}×{size}\n")

if __name__ == "__main__":
    main()
