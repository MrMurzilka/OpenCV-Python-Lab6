"""
Скрипт для исследования алгоритмов компенсации движения.
Вариант 2: Оптический поток (Фарнебак).

Реализует:
- вычисление плотного оптического потока Farneback;
- визуализацию векторов движения на кадрах;
- сравнение нескольких наборов параметров алгоритма.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def default_parameter_sets():
    """Возвращает набор параметров для сравнения разных режимов оптического потока."""
    return [
        {
            "name": "base",
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 15,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0,
        },
        {
            "name": "sensitive",
            "pyr_scale": 0.5,
            "levels": 4,
            "winsize": 21,
            "iterations": 5,
            "poly_n": 7,
            "poly_sigma": 1.5,
            "flags": 0,
        },
        {
            "name": "fast",
            "pyr_scale": 0.6,
            "levels": 2,
            "winsize": 9,
            "iterations": 2,
            "poly_n": 5,
            "poly_sigma": 1.1,
            "flags": 0,
        },
    ]


def compute_optical_flow(
    prev_gray,
    curr_gray,
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
):
    """Вычисляет плотный оптический поток между кадрами методом Фарнебака."""
    if prev_gray.shape != curr_gray.shape:
        raise ValueError(
            "Размеры предыдущего и текущего кадров должны совпадать: "
            f"{prev_gray.shape} vs {curr_gray.shape}"
        )

    flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
    cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        flow,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags,
    )
    return flow


def visualize_motion_vectors(img, flow, step=16, scale=2.0, min_magnitude=1.0, color=(0, 255, 0)):
    """Наносит векторы движения на изображение в виде стрелок."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result = img.copy()
    h, w = img.shape[:2]

    for y in range(0, h, step):
        for x in range(0, w, step):
            dx = flow[y, x, 0]
            dy = flow[y, x, 1]
            magnitude = np.hypot(dx, dy)

            if magnitude < min_magnitude:
                continue

            end_x = int(x + dx * scale)
            end_y = int(y + dy * scale)
            end_x = max(0, min(end_x, w - 1))
            end_y = max(0, min(end_y, h - 1))

            cv2.arrowedLine(
                result,
                (x, y),
                (end_x, end_y),
                color,
                1,
                tipLength=0.35,
            )

    return result


def process_video_with_params(video_path, output_path, params, step=16, scale=2.0, max_frames=None):
    """Обрабатывает всё видео для конкретного набора параметров."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видеофайл: {video_path}")

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError(f"Не удалось прочитать первый кадр из файла: {video_path}")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    h, w = prev_frame.shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Не удалось создать выходной видеофайл: {output_path}")

    print(f"Обработка параметров {params['name']}: {params}")
    processed_frames = 0
    while cap.isOpened():
        if max_frames is not None and processed_frames >= max_frames:
            break

        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = compute_optical_flow(
            prev_gray,
            curr_gray,
            pyr_scale=params["pyr_scale"],
            levels=params["levels"],
            winsize=params["winsize"],
            iterations=params["iterations"],
            poly_n=params["poly_n"],
            poly_sigma=params["poly_sigma"],
            flags=params["flags"],
        )

        visualized_frame = visualize_motion_vectors(curr_frame, flow, step=step, scale=scale)
        out.write(visualized_frame)
        prev_gray = curr_gray
        processed_frames += 1

    cap.release()
    out.release()
    print(f"Готово: {output_path}")


def analyze_motion_optical_flow(video_path, output_dir="results", parameter_sets=None, max_frames=None):
    """Запускает сравнение оптического потока для нескольких наборов параметров."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл видео не найден: {video_path}")

    if parameter_sets is None:
        parameter_sets = default_parameter_sets()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_files = []
    for params in parameter_sets:
        output_file = out_dir / f"optical_flow_{params['name']}.mp4"
        process_video_with_params(str(path), str(output_file), params, max_frames=max_frames)
        output_files.append(output_file)

    return output_files


if __name__ == "__main__":
    video_file = "test_video1.mp4"

    try:
        files = analyze_motion_optical_flow(video_file, output_dir="results")
        print("Сформированы файлы:")
        for file in files:
            print(f"- {file}")
    except Exception as exc:
        print(f"Ошибка выполнения: {exc}")