import cv2
import numpy as np


def center_crop_and_resize(img, crop_size, out_size):
    h, w = img.shape[:2]
    center_h, center_w = h // 2, w // 2
    half = crop_size // 2
    top = max(center_h - half, 0)
    left = max(center_w - half, 0)
    cropped = img[top : top + crop_size, left : left + crop_size]
    resized = cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return resized


def blend_overlay(background, overlay, alpha):
    comp = background.copy().astype(np.float32)
    h_bg, w_bg = comp.shape[:2]
    h_ov, w_ov = overlay.shape[:2]
    top = (h_bg - h_ov) // 2
    left = (w_bg - w_ov) // 2

    roi = comp[top : top + h_ov, left : left + w_ov]
    overlay_f = overlay.astype(np.float32)
    blended_roi = (1 - alpha) * roi + alpha * overlay_f
    comp[top : top + h_ov, left : left + w_ov] = blended_roi
    return comp.astype(np.uint8)


# TODO test using interpolated frames for training Gaussians
# TODO estimate camera poses for interpolated images
def interp_zoomed_images(images, zoom_factor=2.0, fps=30, frames_per_transition=60):
    """
    Params:
        images: List[array[H, W, C]] of dtype uint8 in range [0, 255]
    """
    num_transitions = len(images) - 1
    total_frames = num_transitions * frames_per_transition

    imgs_interp = []
    for f in range(total_frames):
        T = f / float(frames_per_transition)  # T in [0, num_transitions)
        i = int(T)
        t = T - i  # local time in [0, 1]

        s = 1.0 + (zoom_factor - 1.0) * t  # background zoom scale

        crop_size = int(round(1024 / s))
        background = center_crop_and_resize(images[i], crop_size, 1024)
        overlay_size = int(round((1024 * s) / zoom_factor))
        overlay = cv2.resize(
            images[i + 1], (overlay_size, overlay_size), interpolation=cv2.INTER_LINEAR
        )
        frame_img = blend_overlay(background, overlay, alpha=t)
        imgs_interp.append(frame_img)

    hold_frames = int(0.5 * fps)
    for _ in range(hold_frames):
        imgs_interp.append(images[-1])

    return imgs_interp
