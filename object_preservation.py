import numpy as np
import cv2
from seam_carve import *
from tqdm import tqdm

## 適用：只有一維度改變
def object_preserve(img: np.ndarray, mask: np.ndarray, new_shape: tuple([int, int]), energy_mode: str, energy_window: int) -> np.ndarray:
    """object_removal()
    Input:
    - _img: 原圖 (3 channel)
    - mask: 要保留的地方
    - new_shape: (h, w)
    - energy_mode: 'HOG', 'L1', 'L2', 'entropy'
    - energy_window: 只適用於 HOG, entropy
    Output:
    - new_img: 縮小的圖片
    """

    ### 紀錄原圖與新圖的高與寬 ###
    ori_h, ori_w, _ = img.shape
    new_h, new_w = new_shape
    rimg = img.copy()

    seam_num = np.abs(new_h - ori_h) if (new_h - ori_h != 0) else np.abs(new_w - ori_w)       
    seam_orient = "h" if new_h != ori_h else "v"

    #### 1. 進行 seam removal
    # seam_orient = 'h' if ori_w < ori_h else 'v' # 若圖片的寬比較長, 則找水平方向的 seam

    for _ in tqdm(range(seam_num)): 
        ### 計算能量值
        energy_map = energy_function(rimg, energy_mode, energy_window)   
        energy_map = np.where(mask == 255, energy_map * 10, energy_map) # 將遮罩範圍改為 MASK_VALUE
        
        ### 找 seam
        seam_map = find_seam(energy_map, 1, seam_orient) 
        seam_map_3 = np.repeat(seam_map, 3, axis = 1)
        seam_map_3 = seam_map_3.reshape(rimg.shape[0], rimg.shape[1], 3)

        ### remove seam: 做圖片縮減以及 mask 縮減
        rimg = rimg.astype('float32')
        mask = mask.astype('float32')
        np.copyto(rimg, seam_map_3, where = (seam_map_3 == -1)) # 將 seam map 上為 -1 的數值複製至 img 對應位置（將來會被移除）
        np.copyto(mask, seam_map, where = (seam_map == -1))
       
        if (seam_orient == "h"): # 假設是找 horizontal seam，代表是高度有縮減, 而寬度固定
            rimg = cv2.rotate(rimg, cv2.ROTATE_90_CLOCKWISE)
            rimg = rimg[~(rimg == -1)].reshape(rimg.shape[0], -1, 3) 
            rimg = cv2.rotate(rimg, cv2.ROTATE_90_COUNTERCLOCKWISE)

            mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            mask = mask[~(mask == -1)].reshape(mask.shape[0], -1) 
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        else: # 反之為寬度有縮減, 而高度固定
            rimg = rimg[~(rimg == -1)].reshape(rimg.shape[0], -1, 3)
            mask = mask[~(mask == -1)].reshape(mask.shape[0], -1)  

        rimg = rimg.astype('uint8')
        mask = mask.astype('uint8')
    
    return rimg
