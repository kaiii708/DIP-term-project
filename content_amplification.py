import numpy as np
import cv2
from seam_carve import *

def content_amp(img: np.ndarray, energy_mode: str, energy_window: int) -> np.ndarray:
    """content amplification
    Input:
    - img: 原圖 (3 channel)
    - energy_mode: 用於 energy_function, 可選擇 'L1', 'L2', 'entropy', 'HOG', 'forward'
    - window_size: 窗格大小，只有在 mode 為 entropy 或是 HOG 時作用
    Output:
    - new_img: 圖片 (3 channel)
    """
    #######################################
    ### Step 1: 先將原圖放大（邊長的k倍）   ###
    #######################################

    ### 紀錄原圖與新圖的高與寬 ###
    h, w, _ = img.shape
    k = 1.3
    temp_h, temp_w = int(h * k), int(w * k)
    temp_img = cv2.resize(img, (temp_w, temp_h), interpolation = cv2.INTER_CUBIC) # 放大建議使用 INTER_CUBIC
    # temp_img = seam_carve(img, (temp_h, temp_w), energy_mode, energy_window, True)
    print(img.shape, temp_img.shape)

    #######################################
    ### Step 2: 將圖片縮小回原圖大小       ###
    #######################################
    
    new_img = seam_carve(temp_img, (h, temp_w), energy_mode, energy_window, False)
    new_img = seam_carve(new_img, (h, w), energy_mode, energy_window, False)

    return new_img
