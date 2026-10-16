import cv2
import numpy as np


def filtro_pixel_art(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]

    scale = 0.12

    small = cv2.resize(
        roi,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )

    pixel = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    return pixel


def filtro_2(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cell = 6
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cx = (xx % cell) - cell / 2
    cy = (yy % cell) - cell / 2
    dist_center = np.sqrt(cx ** 2 + cy ** 2)
    radius = (1 - gray / 255.0) * (cell / 1.4)
    dot_mask = dist_center < radius
    out = np.full_like(roi, 245)
    out[dot_mask] = (15, 15, 15)
    return out


def filtro_3(roi: np.ndarray) -> np.ndarray:
    shift = 6
    b, g, r = cv2.split(roi)
    r_shift = np.roll(r, -shift, axis=1)
    b_shift = np.roll(b, shift, axis=1)
    out = cv2.merge([b_shift, g, r_shift])
    out[::3, :, :] = (out[::3, :, :] * 0.72).astype(np.uint8)
    return out


def filtro_5(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def filtro_pencil_sketch(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    inverted = 255 - gray

    blur = cv2.GaussianBlur(inverted, (21, 21), 0)

    sketch = cv2.divide(gray, 255 - blur, scale=256)

    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def filtro_blanco(roi: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(roi, (35, 35), 0)
    white = np.full_like(roi, 255)
    out = cv2.addWeighted(blurred, 0.55, white, 0.45, 0)
    return out


def filtro_comic_book(roi: np.ndarray) -> np.ndarray:
    color = roi.copy()

    for _ in range(2):
        color = cv2.bilateralFilter(color, 9, 75, 75)

    Z = color.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0,
    )

    K = 8

    _, label, center = cv2.kmeans(
        Z,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    center = np.uint8(center)

    quantized = center[label.flatten()]
    quantized = quantized.reshape(color.shape)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        8,
    )

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    comic = cv2.bitwise_and(quantized, edges)

    return comic


def filtro_grid(roi: np.ndarray) -> np.ndarray:
    out = roi.copy()
    h, w = out.shape[:2]
    step = 22
    color = (235, 235, 235)

    overlay = out.copy()
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w, y), color, 1)

    out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0)
    return out
def filtro_cartoon(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )

    color = cv2.bilateralFilter(roi, 9, 250, 250)

    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return cv2.bitwise_and(color, edges)

def filtro_rainbow(roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hsv[:, :, 0] = (hsv[:, :, 0] + 35) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def filtro_neon(roi: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(roi, (0, 0), 10)

    glow = cv2.addWeighted(roi, 1.4, blur, -0.5, 0)

    hsv = cv2.cvtColor(glow, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
def filtro_crystal(roi: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(roi, (11, 11), 0)

    crystal = cv2.addWeighted(roi, 0.7, blur, 0.3, 20)

    b, g, r = cv2.split(crystal)

    b = cv2.add(b, 40)
    g = cv2.add(g, 20)

    return cv2.merge([b, g, r])
def filtro_candy(roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.25, 0, 255).astype(np.uint8)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    b, g, r = cv2.split(img)

    r = cv2.add(r, 25)
    b = cv2.add(b, 15)

    return cv2.merge([b, g, r])
def filtro_watercolor(roi: np.ndarray) -> np.ndarray:
    img = cv2.stylization(
        roi,
        sigma_s=60,
        sigma_r=0.45
    )

    return img
def filtro_colorsplash(roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.2, 0, 255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.3, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

FILTER_SETS = [

    (
        filtro_grid,
        filtro_3,
        filtro_2,
    ),

    (
        filtro_pixel_art,
        filtro_comic_book,
        filtro_cartoon,
    ),

    (
        filtro_rainbow,
        filtro_neon,
        filtro_crystal,
    ),

    (
        filtro_candy,
        filtro_watercolor,
        filtro_colorsplash,
    ),

]
FILTROS = [
    filtro_grid,
    filtro_pixel_art,
    filtro_2,
    filtro_3,
    filtro_5,
    filtro_pencil_sketch,
    filtro_blanco,
    filtro_comic_book,
]