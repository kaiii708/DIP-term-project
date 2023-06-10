from find_seam import *

# for trying: call"seam_removal_2d_dp"
# example:"result,_,_=seam_removal_2d_dp(img,2,2,"L2",0,1)"


def seam_removal_2d_dp(_img: np.ndarray, seam_horizontal:int, seam_vertical:int, energy_mode: str, energy_window: int, show: bool)->tuple([np.ndarray,list,list]):
    img=_img.copy()
    energy_map = energy_function(img,energy_mode,energy_window) # 計算能量值
    map_records=find_seam_2d_dp(energy_map,seam_horizontal,seam_vertical)
    seam_records = []
    for ori,seam_map in map_records:
        h,w,_=img.shape

        seam_index = np.where(seam_map == -1)[1] if ori == "v" else np.where(seam_map == -1)[0]
        seam_records.append(seam_index)

        if show: show_seam(img, 'remove', seam_map)

        img = img.astype('float32')
        seam_map_3 = np.repeat(seam_map, 3, axis=1)
        seam_map_3 = seam_map_3.reshape(h, w, 3)
        np.copyto(img, seam_map_3, where = (seam_map_3 == -1)) # 將 seam map 上為 -1 的數值複製至 img 對應位置（將來會被移除）
        
        if (ori == "h"): # 假設是找 horizontal seam，代表是高度有縮減, 而寬度固定
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = img[~(img == -1)].reshape(img.shape[0], -1, 3) 
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        else: # 反之為寬度有縮減, 而高度固定
            img = img[~(img == -1)].reshape(img.shape[0], -1, 3) 
        img = img.astype('uint8')
    return img, map_records, seam_records