"""
Вспомогательные функции для ЛР3
Гауссово размытие изображений
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple


def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Создание матрицы ядра Гаусса
    
    Параметры:
        size (int): Размер ядра (должен быть нечётным)
        sigma (float): Среднеквадратичное отклонение (параметр размытия)
    
    Возвращает:
        np.ndarray: Матрица ядра Гаусса размером size × size
    """
    if size % 2 == 0:
        raise ValueError("Размер ядра должен быть нечётным!")
    
    # Координаты центра матрицы (математическое ожидание)
    center = size // 2
    a, b = center, center
    
    # Создаём пустую матрицу
    kernel = np.zeros((size, size), dtype=np.float64)
    
    # Заполняем матрицу значениями функции Гаусса
    for x in range(size):
        for y in range(size):
            # Формула: gauss[x,y] = (1 / (2*π*σ²)) * e^(-(x-a)² + (y-b)²) / (2*σ²))
            exponent = -((x - a)**2 + (y - b)**2) / (2 * sigma**2)
            kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(exponent)
    
    return kernel


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    """
    Нормализация ядра (сумма элементов = 1)
    
    Параметры:
        kernel (np.ndarray): Ненормированное ядро
    
    Возвращает:
        np.ndarray: Нормированное ядро
    """
    kernel_sum = np.sum(kernel)
    if kernel_sum == 0:
        raise ValueError("Сумма элементов ядра равна нулю!")
    
    normalized = kernel / kernel_sum
    return normalized


def apply_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Применение операции свёртки к изображению
    
    Параметры:
        image (np.ndarray): Исходное изображение (grayscale)
        kernel (np.ndarray): Ядро свёртки
    
    Возвращает:
        np.ndarray: Результат свёртки
    """
    if len(image.shape) != 2:
        raise ValueError("Изображение должно быть в grayscale формате!")
    
    h, w = image.shape
    k_size = kernel.shape[0]
    k_half = k_size // 2
    
    # Создаём копию изображения для результата
    result = np.zeros_like(image, dtype=np.float64)
    
    # Применяем свёртку только к внутренним пикселям
    for i in range(k_half, h - k_half):
        for j in range(k_half, w - k_half):
            # Вырезаем окрестность пикселя
            neighborhood = image[i - k_half:i + k_half + 1, 
                               j - k_half:j + k_half + 1]
            
            # Применяем формулу свёртки: val = Σ Σ B[k,l] * ker[k,l]
            val = np.sum(neighborhood * kernel)
            
            # Ограничиваем значение диапазоном [0, 255]
            result[i, j] = np.clip(val, 0, 255)
    
    return result.astype(np.uint8)


def print_kernel(kernel: np.ndarray, title: str = "Kernel"):
    """
    Красивый вывод матрицы ядра в консоль
    
    Параметры:
        kernel (np.ndarray): Матрица ядра
        title (str): Заголовок
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Размер: {kernel.shape[0]}×{kernel.shape[1]}")
    print(f"Сумма элементов: {np.sum(kernel):.10f}")
    print(f"\nМатрица:")
    
    # Форматированный вывод
    for row in kernel:
        print("  ", end="")
        for val in row:
            print(f"{val:12.8f}", end=" ")
        print()
    
    print(f"{'='*60}\n")


def save_kernel_to_file(kernel: np.ndarray, filename: str):
    """
    Сохранение матрицы ядра в текстовый файл
    
    Параметры:
        kernel (np.ndarray): Матрица ядра
        filename (str): Путь к файлу
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Размер: {kernel.shape[0]}×{kernel.shape[1]}\n")
        f.write(f"Сумма элементов: {np.sum(kernel):.10f}\n\n")
        
        for row in kernel:
            for val in row:
                f.write(f"{val:12.8f} ")
            f.write("\n")
    
    print(f"✓ Ядро сохранено в {filename}")


def compare_images(original: np.ndarray, filtered: np.ndarray, 
                  title: str = "Comparison") -> np.ndarray:
    """
    Создание изображения для сравнения оригинала и результата
    
    Параметры:
        original (np.ndarray): Оригинальное изображение
        filtered (np.ndarray): Отфильтрованное изображение
        title (str): Заголовок
    
    Возвращает:
        np.ndarray: Объединённое изображение
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    
    # Создаём фигуру
    fig = Figure(figsize=(12, 6))
    canvas = FigureCanvasAgg(fig)
    
    # Оригинал
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original', fontsize=14)
    ax1.axis('off')
    
    # Результат
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(filtered, cmap='gray')
    ax2.set_title(f'Filtered: {title}', fontsize=14)
    ax2.axis('off')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    
    # Преобразуем в numpy array
    canvas.draw()
    buf = canvas.buffer_rgba()
    result = np.asarray(buf)
    
    return cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)


def calculate_psnr(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Расчёт PSNR (Peak Signal-to-Noise Ratio) между двумя изображениями
    
    Параметры:
        original (np.ndarray): Оригинальное изображение
        filtered (np.ndarray): Отфильтрованное изображение
    
    Возвращает:
        float: Значение PSNR в dB
    """
    mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
