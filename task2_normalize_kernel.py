"""
ЛАБОРАТОРНАЯ РАБОТА №3 - ЗАДАНИЕ 2
Нормализация матрицы Гаусса
"""

import numpy as np
from utils import create_gaussian_kernel, normalize_kernel, print_kernel, save_kernel_to_file

def main():
    print("ЗАДАНИЕ 2: Нормализация матрицы Гаусса")
    print("=" * 60)
    
    # Те же размеры, что и в задании 1
    sizes = [3, 5, 7]
    sigma = 1.0
    
    for size in sizes:
        print(f"\n Размер ядра: {size}×{size}")
        
        # 1. Создаём НЕНОРМИРОВАННОЕ ядро
        raw_kernel = create_gaussian_kernel(size, sigma)
        print_kernel(raw_kernel, f"Ненормированное ядро {size}×{size}")
        
        # 2. Нормируем ядро
        norm_kernel = normalize_kernel(raw_kernel)
        print_kernel(norm_kernel, f"НОРМИРОВАННОЕ ядро {size}×{size}")
        
        # Проверяем сумму = 1.0
        assert abs(np.sum(norm_kernel) - 1.0) < 1e-10, "Нормализация неверна!"
        print("✓ Нормализация проверена (сумма = 1.0000000000)")
        
        # Сохраняем
        filename = f"output/task2_kernel_norm_{size}x{size}_sigma{sigma}.txt"
        save_kernel_to_file(norm_kernel, filename)
        
        print(f"✓ Обработано размер {size}×{size}\n")

if __name__ == "__main__":
    main()
