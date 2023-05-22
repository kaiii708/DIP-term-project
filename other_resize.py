import numpy as np
import cv2
from energy_function import *

def crop(img: np.ndarray, new_shape: tuple([int, int]), energy_mode: str, energy_window: int) -> np.ndarray:
    """crop: 找到能量總和最大的方框
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


    energy_map = energy_function(img, energy_mode, energy_window)

    max_xy = (0, 0)
    max_energy = 0
    for i in range(ori_h - new_h):
        for j in range(ori_w - new_w):
            energy = energy_map[i : i + new_h, j : j + new_w]
            if (energy > max_energy):
                max_xy = (i, j)
                max_energy = energy
    
    return img[max_xy[0] : max_xy[0] + new_h, max_xy[1] : max_xy[1] + new_w].astype("uint8")

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

    rimg = np.zeros((new_h, new_w, 3))
    if(new_h < ori_h and new_w == ori_w):
        for c in range(ori_w):
            remove_idx = col_sort_idx[:, c][:(ori_h - new_h)]
            cur_col = img[:, c, :].copy()
            cur_col = np.delete(cur_col, remove_idx, axis = 0)
            rimg[:,c ,:] = cur_col
    
    elif(new_w < ori_w and new_h == ori_h):
        for r in range(ori_h):
            remove_idx = row_sort_idx[r, :][:(ori_w - new_w)]
            cur_row = img[r, :, :].copy()
            cur_row = np.delete(cur_row, remove_idx, axis = 0)
            rimg[r,: ,:] = cur_row

    elif(new_h < ori_h and new_w < ori_w and row_first == True):
        pre_rimg = remove_from_each_row_col(img, (new_h, ori_w), True, energy_mode, energy_window)
        rimg = remove_from_each_row_col(pre_rimg, (new_h, new_w), True, energy_mode, energy_window)

    elif(new_h < ori_h and new_w < ori_w and row_first == False):
        pre_rimg = remove_from_each_row_col(img, (ori_h, new_w), False, energy_mode, energy_window)
        rimg = remove_from_each_row_col(pre_rimg, (new_h, new_w), False, energy_mode, energy_window)
    else:
        rimg = img.copy()

    return rimg.astype("uint8")

def other_resize_method(img: np.ndarray, new_shape: tuple([int, int]), resize_method: str, row_first: bool, energy_mode: str, energy_window: int) -> np.ndarray:
    """other_resize_method: 其他縮減、放大方式
    Input:
    - img: 原圖 (3 channels)
    - new_shape: (new_h, new_w)
    - resize_method: 'crop', 'remove_row_col', 'remove_from_each_row_col', 'scale'。只有 scale 適用於放大與縮小，前三者只可以縮減
    - row_first: 在 remove_row_col, remove_from_each_row_col 有用，指定是否要由列開始處理
    - energy_mode: 'entropy', 'L1', 'L2', 'HOG', 'entropy', 'forward'
    - energy_window: 窗格，只適用在 'HOG', 'entropy'
    Output:
    - rimg: 改變大小後圖片
    """    
    if resize_method == 'crop':
        rimg = crop(img, new_shape, energy_mode, energy_window)
    elif resize_method == 'remove_row_col':
        rimg = remove_row_col(img, new_shape, row_first, energy_mode, energy_window)
    elif resize_method == 'remove_from_each_row_col':
        rimg = remove_row_col(img, new_shape, row_first, energy_mode, energy_window)
    elif resize_method == 'scale':
        rimg = scale(img, new_shape)
    else:
        raise ValueError("沒有這個指定模式！請選擇：'crop', 'remove_row_col', 'remove_from_each_row_col', 'scale'")
    
    return rimg
