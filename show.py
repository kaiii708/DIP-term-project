
import numpy as np
import cv2
import time

def show_seam(img, mode, seam_map=None):
    if mode == 'insert':
        time.sleep(1)
    vis = img.copy()
    if seam_map is not None:
        vis[np.where(seam_map == -1)] = [0,0,255]
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis