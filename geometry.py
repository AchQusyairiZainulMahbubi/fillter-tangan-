import cv2
import numpy as np


# def portal_width(p1, p2, p3, p4):
#     top_w = np.hypot(p3[0] - p1[0], p3[1] - p1[1])
#     bottom_w = np.hypot(p4[0] - p2[0], p4[1] - p2[1])
#     return (top_w + bottom_w) / 2.0


# class ClosingGestureDetector:

#     def __init__(self, close_ratio=0.16, open_ratio=0.30):
#         self.close_ratio = close_ratio
#         self.open_ratio = open_ratio
#         self.is_closed = False

#     def update(self, width, frame_w):
#         close_threshold = self.close_ratio * frame_w
#         open_threshold = self.open_ratio * frame_w

#         triggered = False
#         if not self.is_closed and width < close_threshold:
#             self.is_closed = True
#             triggered = True
#         elif self.is_closed and width > open_threshold:
#             self.is_closed = False

#         return triggered


def paint_filter_in_polygon(frame, polygon_pts, filtro_func):
    h, w = frame.shape[:2]

    x, y, bw, bh = cv2.boundingRect(polygon_pts)
    x, y = max(x, 0), max(y, 0)
    bw = min(bw, w - x)
    bh = min(bh, h - y)
    if bw <= 1 or bh <= 1:
        return frame

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts], 255)
    mask_roi = mask[y : y + bh, x : x + bw]

    roi = frame[y : y + bh, x : x + bw]
    filtered_roi = filtro_func(roi)

    mask_3ch = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    blended = (filtered_roi * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
    frame[y : y + bh, x : x + bw] = blended
    return frame


def render_portal(frame, p1, p2, p3, p4, filtro_func):
    full_polygon = np.array([p1, p3, p4, p2], dtype=np.int32)

    paint_filter_in_polygon(frame, full_polygon, filtro_func)

    cv2.polylines(frame, [full_polygon], isClosed=True, color=(255, 255, 255), thickness=1)
    return frame