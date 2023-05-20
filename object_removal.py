import numpy as np
import cv2
from seam_carve import *

MASK_VALUE = -999999
pts_list = [] # 存取使用者的點

def click_event(event, x, y, flags, params):
    """透過滑鼠點擊決定遮罩的角點"""
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 1, (0, 0, 255), 2) # 在點擊處畫上紅點
        pts_list.append([x, y])
        cv2.imshow("get mask", img)

def get_mask(img: np.ndarray) -> np.ndarray:
    """get_mask
    Input:
    - img: 原圖（3 channels）
    Output:
    - mask: 1 channel
    """
   # img = _img.copy()
    cv2.imshow("get mask", img)
    cv2.setMouseCallback('get mask', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    pts_arr = np.array(pts_list, np.int32).reshape((-1, 1, 2))
    
	# 將遮罩顯示於原圖
    cv2.fillPoly(img, [pts_arr], (0, 0, 255))
    cv2.imshow("get mask", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   

	# 建立遮罩
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [pts_arr], (0, 0, 255))
    mask = (np.sum(mask, axis = 2) / 255).astype("uint8")
    
    return mask

def object_removal(_img: np.ndarray, seam_orient: str, energy_mode: str, energy_window: int) -> np.ndarray:
    """object_removal()
    Input:
    - _img: 原圖 (3 channel)
    - seam_orient: 方向, 'h' or 'v'
    - energy_mode: 'HOG', 'L1', 'L2', 'entropy'
    - energy_window: 只適用於 HOG, entropy
    Output:
    - new_img: 移除物體以後的圖片
    """
    ### 0. 前置作業：取得 mask，並確認大小關係、方向正確。
    mask_img = _img.copy()
    rimg = _img.copy().astype("float32")
    mask = get_mask(mask_img).astype("float32") # 取得 mask

    ori_h, ori_w, _ = _img.shape # 原圖高、寬
    mask_h, mask_w = mask.shape # mask 的高、寬
    if (ori_h != mask_h or ori_w != mask_w): # 確認
        raise ValueError("原圖和遮罩的高度與寬度需要相同！")
    
    if(seam_orient not in ['v', 'h']):
        raise ValueError("seam_orient can only be h or v.")

    #### 1. 進行 seam removal
    # seam_orient = 'h' if ori_w < ori_h else 'v' # 若圖片的寬比較長, 則找水平方向的 seam

    while(np.sum(mask) > 0): # 持續執行至所有遮罩範圍都被移除
        # print(np.sum(mask))
        ### 計算能量值
        energy_map = energy_function(rimg, energy_mode, energy_window)   
        energy_map = np.where(mask == 1, MASK_VALUE, energy_map) # 將遮罩範圍改為 MASK_VALUE
        
        ### 找 seam
        seam_map = find_seam(energy_map, 1, seam_orient) 
        seam_map_3 = np.repeat(seam_map, 3, axis = 1)
        seam_map_3 = seam_map_3.reshape(rimg.shape[0], rimg.shape[1], 3)

        ### remove seam: 做圖片縮減以及 mask 縮減
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

        # cv2.imwrite("object_removal.png", rimg)
    
    # seam insertion
    result_img = seam_carve(rimg, (ori_h, ori_w), energy_mode, energy_window, True)

    return result_img

# img = cv2.imread("test.JPG")
# result_img = object_removal(img, 'v', 'L1', 3)
# cv2.imwrite("result.png", result_img)
