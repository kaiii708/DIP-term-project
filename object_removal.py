import numpy as np
import cv2
from seam_carve import *

MASK_VALUE = -9999999

def object_removal(_img: np.ndarray, mask_img: np.ndarray, energy_mode: str, energy_window: int) -> np.ndarray:
    """object_removal
    Input:
    - _img: 原圖 (3 channel)
    - mask_img: 寬與高與原圖相同 (2 channel), 1 代表要移除的地方, 0 則是保留
    - energy_mode: 用於決定使用哪種方式計算 energy, 可以選擇 HOG, entropy, L1, L2
    - energy_window: 只適用於 HOG, entropy
    Output:
    - new_img: 移除完後的圖片 (3 channel)
    """
    #### Step 1: 前置工作
    # 確認 mask 與原圖大小相同
    ori_h, ori_w, _ = img.shape
    mask_h, mask_w = mask_img.shape

    if (ori_h != mask_h or ori_w != mask_w):
        raise ValueError("原圖和遮罩的高度與寬度需要相同！")
    
    # 將遮罩的元素轉換為足夠大的負值, 如此一來選擇 seam 時會優先選擇
    mask_img = np.where(mask_img == 1, MASK_VALUE, 0).astype("float64")

    # 複製一份原圖，並且轉為浮點數型態
    img = _img.astype("float64").copy()

    ##### Step 2: 進行 seam removal 直至所有需遮罩的元素被移除

    # 決定要移除的方向
    seam_orient = 'h' if ori_w < ori_h else 'v' # 若圖片的寬比較長, 則找水平方向的 seam

    # 重複執行 seam removal 直至所有遮罩範圍被移除
    while(np.sum(mask_img) > 0): 

        energy_map = energy_function(img, energy_mode, energy_window)
        seam_map = find_seam(energy_map, 1, seam_orient)
        remove_img = np.zeros_like(img)   

        # 更新 img
        seam_map_3 = np.repeat(seam_map, 3, axis = 1)
        seam_map_3 = seam_map_3.reshape(ori_h, ori_w, 3)
        np.copyto(img, seam_map_3, where = (seam_map_3 == -1)) # 將 seam map 上為 -1 的數值複製至 img 對應位置（將來會被移除）
        
        if (seam_orient == "h"): # 假設是找 horizontal seam，代表是高度有縮減, 而寬度固定
            img = img[~(img == -1)].reshape(-1, img.shape[1], 3)
        else: # 反之為寬度有縮減, 而高度固定
            img = img[~(img == -1)].reshape(img.shape[0], -1, 3) 
        
        # 更新 seam map
        if (seam_orient == "h"):
            seam_map = seam_map[~(seam_map == -1)].reshape(-1, img.shape[1])
        else:
            seam_map = seam_map[~(seam_map == -1)].reshape(img.shape[0], -1)

    ##### Step 3: 進行 seam insertion 使得圖片大小與原圖相同
    ins_img = seam_carve(remove_img, (ori_h, ori_w), energy_mode, energy_window)

    return ins_img