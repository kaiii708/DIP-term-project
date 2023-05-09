import cv2
import numpy as np
import time
from tqdm import tqdm
from energy_function import *
from find_seam import *
from remove_seam import *


def add_seam(img: np.ndarray,  seam_index: np.ndarray, seam_orient: str):
    """ add_seam
    Input:
    - img: 圖片 (3 channel)
    - seam_index: 要插入的 seam 的座標
    - seam_orient: 找尋的 seam 方向：垂直-'v', 水平-'h'
    Output:
    - new_img: 插入 seam 以後的圖片 (3 channel)
    """

    h, w, _ = img.shape

    # Create a new image with new shape
    new_img = np.zeros((h,w+1,3)) if seam_orient == 'v' else np.zeros((h+1, w,3))

    # Insert vertical seam
    if seam_orient == 'v':
        for row in range(h):
            col = seam_index[row]
            if col > w : col = w
            if col == 0:
                b = np.average(img[row, col: col + 2, 0])
                g = np.average(img[row, col: col + 2, 1])
                r = np.average(img[row, col: col + 2, 2])
                new_img[row, col, :] = img[row, col, :]
                new_img[row, col + 1, :] = b,g,r
                # new_img[row, col + 1, :] = [0,0,255]
                new_img[row, col + 1:, :] = img[row, col:, :]
            else:
                b = np.average(img[row, col - 1: col + 1, 0])
                g = np.average(img[row, col - 1: col + 1, 1])
                r = np.average(img[row, col - 1: col + 1, 2])
                new_img[row, : col, :] = img[row, : col, :]
                new_img[row, col, :] = b,g,r
                # new_img[row, col, :] = [0,0,255]
                new_img[row, col + 1:, :] = img[row, col:, :]

    # Insert horizontal seam
    else:
        for col in range(w):
            row = seam_index[col]
            if row > h : row = h
            if row == 0:
                b = np.average(img[row: row + 2, col, 0])
                g = np.average(img[row: row + 2, col, 1])
                r = np.average(img[row: row + 2, col, 2])
                new_img[row, col, :] = img[row, col, :]
                new_img[row + 1, col, :] = b,g,r
                # new_img[row + 1, col, :] = [0,0,255]
                new_img[row + 1:, col, :] = img[row:, col, :]
            else:
                b = np.average(img[row - 1: row + 1, col, 0])
                g = np.average(img[row - 1: row + 1, col, 1])
                r = np.average(img[row - 1: row + 1, col, 2])
                new_img[: row, col, :] = img[: row, col, :]
                new_img[row, col, :] = b,g,r
                # new_img[row, col, :] = [0,0,255]
                new_img[row + 1:, col, :] = img[row:, col, :]
        
    new_img = new_img.astype('uint8')    
    return new_img

def seam_insertion(img: np.ndarray,  num_insert: int, seam_orient: str, energy_mode: str, energy_window:int):
    """seam_insertion
    Input:
    - img: 圖片 (3 channel)
    - seam_num: 要插入的 seam 數量
    - seam_orient: 找尋的 seam 方向：垂直-'v', 水平-'h'
    - energy_mode: energy function 模式，可以選擇 'L1', 'L2', 'HOG', 'entropy'
    - energy_window: 窗格大小，只有在 HOG, entropy 有用
    Output:
    - new_img: 插入 seam 以後的圖片 (3 channel)
    """

    seam_records = []
    new_img = img.copy()

    print('Calculating the order of removed seams...')
    for _ in tqdm(range(num_insert)):
        img = img.astype('uint8')
        energy_map = energy_function(img, energy_mode, energy_window)   
        seam_map = find_seam(energy_map, 1, seam_orient)
        seam_index = np.where(seam_map == -1)[1] if seam_orient == "v" else np.where(seam_map == -1)[0]
        img = seam_removal(img, 1, seam_orient, energy_mode, energy_window)
        seam_records.append(seam_index)

    for _ in range(num_insert):
        seam = seam_records.pop()
        new_img = add_seam(new_img, seam, seam_orient)
        for item in seam_records:
            item[np.where(item >= seam)] += 2

    return new_img


