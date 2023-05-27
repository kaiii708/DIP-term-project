import numpy as np
import cv2
from seam_carve import *

R_MASK_VALUE = -999999
P_MASK_VALUE = 999999

def object_removal_with_protective_mask(_img: np.ndarray, p_mask: np.ndarray, r_mask: np.ndarray, seam_orient: str, energy_mode: str, energy_window: int) -> np.ndarray:
    """object_removal()
    Input:
    - _img: 原圖 (3 channel)
    - p_mask: 要保留的部分
    - r_mask: 要移除的部分
    - seam_orient: 方向, 'h' or 'v'
    - energy_mode: 'HOG', 'L1', 'L2', 'entropy', 'forward'
    - energy_window: 只適用於 HOG, entropy
    Output:
    - new_img: 移除物體以後的圖片
    """

    ### 0. 前置作業：取得 mask，並確認大小關係、方向正確。
    rimg = _img.copy()

    ori_h, ori_w, _ = _img.shape # 原圖高、寬
    r_mask_h, r_mask_w = r_mask.shape # r_mask 的高、寬
    p_mask_h, p_mask_w = p_mask.shape # p_mask 的高、寬

    if (ori_h != r_mask_h or ori_w != r_mask_w): # 確認
        raise ValueError("原圖和遮罩的高度與寬度需要相同！")
    if (ori_h != p_mask_h or ori_w != p_mask_w): # 確認
        raise ValueError("原圖和遮罩的高度與寬度需要相同！")
    
    #### 1. 進行 seam removal

    seam_records = []

    while(np.sum(r_mask) > 0): # 持續執行至所有遮罩範圍都被移除
        ### 計算能量值
        energy_map = energy_function(rimg, energy_mode, energy_window)   
        energy_map = np.where(r_mask == 255, R_MASK_VALUE, energy_map) # 將R遮罩範圍改為 R_MASK_VALUE
        energy_map = np.where(p_mask == 255, P_MASK_VALUE, energy_map) # 將P遮罩範圍改為 P_MASK_VALUE
        
        ### 找 seam
        seam_map = find_seam(energy_map, 1, seam_orient) 
        seam_map = find_seam(energy_map, 1, seam_orient) # 求得 seam map
        seam_index = np.where(seam_map == -1)[1] if seam_orient == "v" else np.where(seam_map == -1)[0]
        seam_records.append(seam_index)

        seam_map_3 = np.repeat(seam_map, 3, axis = 1)
        seam_map_3 = seam_map_3.reshape(rimg.shape[0], rimg.shape[1], 3)

        ### remove seam: 做圖片縮減以及 mask 縮減
        rimg = rimg.astype('float32')
        p_mask = p_mask.astype('float32')
        r_mask = r_mask.astype('float32')
        np.copyto(rimg, seam_map_3, where = (seam_map_3 == -1)) # 將 seam map 上為 -1 的數值複製至 img 對應位置（將來會被移除）
        np.copyto(p_mask, seam_map, where = (seam_map == -1))
        np.copyto(r_mask, seam_map, where = (seam_map == -1))
       
        if (seam_orient == "h"): # 假設是找 horizontal seam，代表是高度有縮減, 而寬度固定
            rimg = cv2.rotate(rimg, cv2.ROTATE_90_CLOCKWISE)
            rimg = rimg[~(rimg == -1)].reshape(rimg.shape[0], -1, 3) 
            rimg = cv2.rotate(rimg, cv2.ROTATE_90_COUNTERCLOCKWISE)

            p_mask = cv2.rotate(p_mask, cv2.ROTATE_90_CLOCKWISE)
            p_mask = p_mask[~(p_mask == -1)].reshape(p_mask.shape[0], -1) 
            p_mask = cv2.rotate(p_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            r_mask = cv2.rotate(r_mask, cv2.ROTATE_90_CLOCKWISE)
            r_mask = r_mask[~(r_mask == -1)].reshape(r_mask.shape[0], -1) 
            r_mask = cv2.rotate(r_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        else: # 反之為寬度有縮減, 而高度固定
            rimg = rimg[~(rimg == -1)].reshape(rimg.shape[0], -1, 3)
            p_mask = p_mask[~(p_mask == -1)].reshape(p_mask.shape[0], -1)  
            r_mask = r_mask[~(r_mask == -1)].reshape(r_mask.shape[0], -1)  

        rimg = rimg.astype('uint8')
        p_mask = p_mask.astype('uint8')
        r_mask = r_mask.astype('uint8')

    # seam insertion
    result_img = rimg.copy()
    for _ in tqdm(range(len(seam_records))):
        seam = seam_records.pop()
        points = np.stack((seam, np.arange(0,seam.shape[0],1)), axis=1)
        result_img = add_seam(result_img, seam, seam_orient)
        map = np.zeros((result_img.shape[0], result_img.shape[1]))
        for point in points:
            if seam_orient == 'h':
                if point[0] >= map.shape[1]: 
                    map[map.shape[1] - 1, point[1]] = -1
                else:
                    map[point[0],point[1]] = -1
            else:
                if point[0] >= map.shape[1]: 
                    map[point[1], map.shape[1] - 1] = -1
                else:
                    map[point[1],point[0]] = -1
        for item in seam_records:
            item[np.where(item > seam)] += 2

    return result_img
