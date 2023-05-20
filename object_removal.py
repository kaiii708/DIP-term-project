import numpy as np
import cv2
from seam_carve import *

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

img = cv2.imread("test.JPG")
mask = get_mask(img)
