import numpy as np
import cv2
from find_seam import *
from energy_function import *
from remove_seam import *
from insert_seam import *

def seam_carve(img: np.ndarray, new_shape: tuple([int, int]), energy_mode: str, energy_window: int) -> np.ndarray:
    """seam_carve
    Input:
    - img: 原圖 (3 channel)
    - new_shape: (h, w)
    - energy_mode: 用於 energy_function, 可選擇 'L1', 'L2', 'entropy', 'HOG'
    - window_size: 窗格大小，只有在 mode 為 entropy 或是 HOG 時作用
    Output:
    - new_img: 更改大小後的圖片"""

    ### 紀錄原圖與新圖的高與寬 ###
    ori_h, ori_w = img.shape
    new_h, new_w = new_shape

    ### 根據大小決定要呼叫的函數 ###
    if(ori_h != new_h and ori_w != new_w): # 兩個維度皆改變：將圖片以 max{(新_高 / 舊_高), (新_寬 / 舊_寬)} 作為倍數縮放，接著進行 seam_removal
        ## STEP 1: 縮放圖片
        ratio_h, ratio_w = new_h / ori_h, new_w / ori_w # 得到寬與高原圖與新圖之比例
        if (ratio_h > ratio_w): # 假設高的比例較大，則以其作為基準放大
            if (ratio_h > 1):
                scale_img = cv2.resize(img, (new_h, new_h/ori_w), interpolation = cv2.INTER_CUBIC) # 放大建議使用 INTER_CUBIC
            else:
                scale_img = cv2.resize(img, (new_h, new_h/ori_w), interpolation = cv2.INTER_AREA) # 縮小建議使用 INTER_AREA
        else:
            if (ratio_w > 1):
                scale_img = cv2.resize(img, (new_w/ori_h, new_w), interpolation = cv2.INTER_CUBIC) # 放大建議使用 INTER_CUBIC
            else:
                scale_img = cv2.resize(img, (new_w/ori_h, new_w), interpolation = cv2.INTER_AREA) # 縮小建議使用 INTER_AREA
       
        ## STEP 2: seam_removal
        seam_num = np.abs(new_h - scale_img.shape[0]) if (new_h - scale_img.shape[0] != 0) else np.abs(new_w - scale_img.shape[1])
        seam_orient = "h" if new_h != scale_img else "v" # 若高度不同，則需要找得是 horizontal seam，反之為 vertical seam
        seam_img = seam_removal(img, seam_num, seam_orient, energy_mode, energy_window)

    else: # 只有一個維度改變
        seam_num = np.abs(new_h - ori_h) if (new_h - ori_h != 0) else np.abs(new_w - ori_w)       
        seam_orient = "h" if new_h != ori_h else "v"

        if (new_h < ori_h or new_w < ori_w) # 縮小
            seam_img = seam_removal(img, seam_num, seam_orient, energy_mode, energy_window)
        else: # 放大
            seam_img = seam_insertion(img, seam_num, seam_orient, energy_mode, energy_window)

    return seam_img.astype("uint8")