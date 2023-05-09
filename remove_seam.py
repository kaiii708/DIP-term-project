import numpy as np
from find_seam import *
from energy_function import *

def seam_removal(_img: np.ndarray, seam_num: int, seam_orient: str, energy_mode: str, energy_window: int) -> np.ndarray:
    """seam_removal
    Input:
    - img: 圖片 (3 channel)
    - seam_num: 要移除的 seam 數量
    - seam_orient: 找尋的 seam 方向：垂直-'v', 水平-'h'
    - energy_mode: energy function 模式，可以選擇 'L1', 'L2', 'HOG', 'entropy'
    - energy_window: 窗格大小，只有在 HOG, entropy 有用
    Output:
    - new_img: 移除 seam 以後的圖片 (3 channel)
    """
    
    img = _img.copy() # 先複製一份 img

    for _ in range(seam_num): # 重複操作 seam_num 輪
        h, w, _ = img.shape
        energy_map = energy_function(img, energy_mode, energy_window) # 計算能量值
        seam_map = find_seam(energy_map, 1, seam_orient) # 求得 seam map
        
        img = img.astype('float64')
        seam_map_3 = np.repeat(seam_map, 3, axis=1)
        seam_map_3 = seam_map_3.reshape(h, w, 3)

        np.copyto(img, seam_map_3, where = (seam_map_3 == -1)) # 將 seam map 上為 -1 的數值複製至 img 對應位置（將來會被移除）
       
        if (seam_orient == "h"): # 假設是找 horizontal seam，代表是高度有縮減, 而寬度固定
            img = img[~(img == -1)].reshape(-1, img.shape[1], 3)
        else: # 反之為寬度有縮減, 而高度固定
            img = img[~(img == -1)].reshape(img.shape[0], -1, 3) 

    return img