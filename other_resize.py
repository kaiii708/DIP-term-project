import numpy as np
import cv2
from energy_function import *

def crop(img: np.ndarray, new_shape: tuple([int, int])) -> np.ndarray:
    """crop (center): 保留最中心部分
    Input
    - img: 圖片 (3 channels)
    - new_shape: (h, w)
    Output
    - cropped_img: 裁切後圖片 (3 channels)    
    """
    # 檢查 new_shape < img    
    new_h, new_w, = new_shape
    ori_h, ori_w, _ = img.shape

    if not (new_h <= ori_h and new_w <= ori_w):
        raise ValueError("裁切後圖片尺寸較小")

    # 得到原圖與新圖的中心
    ori_cen_h, ori_cen_w = ori_h // 2, ori_w // 2
    new_cen_h, new_cen_w = new_h // 2, new_w // 2

    cropped_img = img[ori_cen_h - new_cen_h : ori_cen_h + new_cen_h + (new_h % 2), ori_cen_w - new_cen_w : ori_cen_w + new_cen_w +  (new_w % 2)]

    return cropped_img

def remove_row_col(img: np.ndarray, new_shape: tuple([int, int]), row_first: bool, energy_mode: str, energy_window: int) -> np.ndarray:
    """remove_row_col: 移除能量總和最小的列或是欄
    Input:
    - img: 原圖 (3 channels)
    - new_shape: (h, w)    
    - row_fitst: 遇到二維變動時，是否要先移除 row
    - energy_mode: 'HOG', 'L1', 'L2', 'entropy', 'HOG', 'forward'
    - energy_window: 窗格，僅作用在 HOG, entropy
    Output: 
    - rimg: 縮小後的圖片
    """
    # 檢查 new_shape < img    
    new_h, new_w, = new_shape
    ori_h, ori_w, _ = img.shape

    if not (new_h <= ori_h and new_w <= ori_w):
        raise ValueError("裁切後圖片尺寸較小")

    # 得到 energy_map
    energy_map = energy_function(img, energy_mode, energy_window)
    row_sum_sort_idx = np.argsort(np.sum(energy_map, axis = 1)) # 由小至大
    col_sum_sort_idx = np.argsort(np.sum(energy_map, axis = 0))

    # 移除列 or 行
    if(new_h < ori_h and new_w == ori_w): # 若只有高度不同，則要移除列
        rimg = np.delete(img, row_sum_sort_idx[:(ori_h - new_h)], 0) # 移除橫列
    elif(new_w < ori_w and new_h == ori_h): # 若只有寬度不同，則要移除欄
        rimg = np.delete(img, col_sum_sort_idx[:(ori_w - new_w)], 1) # 移除直欄
    elif(new_h < ori_h and new_w < ori_w and row_first == True):
        rimg = np.delete(img, row_sum_sort_idx[:(ori_h - new_h)], 0)
        rimg = np.delete(rimg, col_sum_sort_idx[:(ori_w - new_w)], 1)
    elif(new_h < ori_h and new_w < ori_w and row_first == False):
        rimg = np.delete(img, col_sum_sort_idx[:(ori_w - new_w)], 1)
        rimg = np.delete(rimg, row_sum_sort_idx[:(ori_h - new_h)], 0)
    else:
        rimg = img.copy()

    return rimg.astype("uint8")

def scale(img: np.ndarray, new_shape: tuple([int ,int])) -> np.ndarray:
    """scale: 直接使用 cv2.resize() 改變圖片大小
    Input:
    - img: 原圖 (3 channels)
    - new_shape: (h, w)
    Output:
    - rimg: 改變大小後的圖 (3 channels)    
    """
    rimg = cv2.resize(img, (new_shape[1], new_shape[0]))
    return rimg.astype("uint8") 

def remove_from_each_row_col(img: np.ndarray, new_shape: tuple([int, int]), row_first: bool, energy_mode: str, energy_window: int) -> np.ndarray:
    """remove_from_each_row_col: 從每一列/行刪除最小的幾個像素
    Input:
    - img: 原圖 (3 channels)
    - new_shape: (h, w)    
    - row_fitst: 遇到二維變動時，是否要先移除 row
    - energy_mode: 'HOG', 'L1', 'L2', 'entropy', 'HOG', 'forward'
    - energy_window: 窗格，僅作用在 HOG, entropy
    Output: 
    - rimg: 縮小後的圖片    
    """
    # 檢查圖片大小是否為縮小
    new_h, new_w, = new_shape
    ori_h, ori_w, _ = img.shape

    if not (new_h <= ori_h and new_w <= ori_w):
        raise ValueError("裁切後圖片尺寸較小")
    
    # 計算 energy map
    energy_map = energy_function(img, energy_mode, energy_window)
    row_sort_idx = np.argsort(energy_map, axis = 1) # 由小至大
    col_sort_idx = np.argsort(energy_map, axis = 0)

    # 開始進行移除動作
    rimg = np.zeros(new_shape)
    # if (new_h < ori_h and new_w == ori_w): # 假設高度不同，則各欄位取出最小的幾個像素
    #     for c in range(ori_w):
    #         cur_col = img[:, c].copy()
    #         rimg[:, c] = np.delete(cur_col,)



img = cv2.imread("mypic.JPG")
print(img.shape)

rimg = scale(img, (500, 1000))
print(rimg.shape)

cv2.imshow("img", rimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

