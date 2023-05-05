import cv2
import numpy as np
from numpy import unravel_index
from matplotlib import pyplot as plt
import math
import string

# energy function
# gray_scale
def L1_norm(g_r,g_c):
    return abs(g_r)+abs(g_c)

def L2_norm(g_r,g_c):
    return math.hypot(g_r,g_c)

def entropy(_img: np.ndarray, window_size: int) -> np.ndarray:
    """get_entropy
    Input:
    - _img: input image; it won't be changed inside this function
    - window_size: window size; it should be an odd number
    Output:
    - entropy map: calculate the entropy of each window and fill the value in the corrsponding cell.
    """
    # Check whether window_size is odd
    if (window_size % 2 == 0):
        raise ValueError("window_size should be an odd number.")

    # Pad the image and store it as a new image
    pad_width = int(window_size / 2)
    img = np.pad(_img, (pad_width, pad_width), 'edge')

    # Create an entropy map
    entropy_map = np.zeros(_img.shape, dtype = np.float32)

    # Start to calculate entropy values
    for i in range(_img.shape[0]):
        for j in range(_img.shape[1]):
            window = img[i : i + window_size, j : j + window_size] # obtain the overlapping region
            _, counts = np.unique(window, return_counts = True) # get the counts of each intensity value
            p = counts / (window_size * window_size) # calculate p(x_i)
            entropy_map[i, j] = -np.sum(np.log2(p) * p) # plug in the entropy function
    
    return entropy_map

def HOG(img,win_size):
    height,width=img.shape
    G,O=gradient_magntitue_and_orientation(img,2,1)
    pad_size = win_size//2
    G_pad = np.pad(G,(pad_size,pad_size),'edge')
    O_pad = np.pad(O,(pad_size,pad_size),'edge')
    res = np.zeros(G_pad.shape)
    for i in range(pad_size,pad_size+height):
        for j in range(pad_size,pad_size+width):
            G_window = G_pad[i-pad_size:i+pad_size+1,j-pad_size:j+pad_size+1].ravel()
            O_window = O_pad[i-pad_size:i+pad_size+1,j-pad_size:j+pad_size+1].ravel()
            bins=[0 for i in range(9)]
            for k in range(0,pow(win_size,2)):
                g=G_window[k]
                o=O_window[k]
                if -180<=o<0:
                    o+=180
                if 0<=o<20:
                    bins[0]+=((20-o)*g/20)
                    bins[1]+=((o-0)*g/20)
                elif 20<=o<40:
                    bins[1]+=((40-o)*g/20)
                    bins[2]+=((o-20)*g/20)
                elif 40<=o<60:
                    bins[2]+=((60-o)*g/20)
                    bins[3]+=((o-40)*g/20)
                elif 60<=o<80:
                    bins[3]+=((80-o)*g/20)
                    bins[4]+=((o-60)*g/20)
                elif 80<=o<100:
                    bins[4]+=((100-o)*g/20)
                    bins[5]+=((o-80)*g/20)
                elif 100<=o<120:
                    bins[5]+=((120-o)*g/20)
                    bins[6]+=((o-100)*g/20)
                elif 120<=o<140:
                    bins[6]+=((140-o)*g/20)
                    bins[7]+=((o-120)*g/20)
                elif 140<=o<160:
                    bins[7]+=((160-o)*g/20)
                    bins[8]+=((o-140)*g/20)
                elif 160<=o<180:
                    bins[8]+=((180-o)*g/20)
                    bins[0]+=((o-160)*g/20)
            max_hog=max(bins)
            if max_hog == 0:
                max_hog=1
            res[i][j]=G_pad[i][j]/max_hog
    return res[pad_size:pad_size+height,pad_size:pad_size+width]

# by sobel, K=2
# L1-norm:norm=1,L2-norm:norm=2
def gradient_magntitue_and_orientation(img,K,norm):
    height,width = img.shape
    row_grad_mask =np.array([[-1,0,1],[-K,0,K],[-1,0,1]])/(K+2)
    col_grad_mask =np.array([[1,K,1],[0,0,0],[-1,-K,-1]])/(K+2)
    kernel_size=row_grad_mask.shape[0]
    pad_size = kernel_size//2
    # pad_img = cv2.copyMakeBorder(img,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REFLECT)
    pad_img = np.pad(img,(pad_size,pad_size),'edge')
    G= np.zeros(pad_img.shape)
    O= np.zeros(pad_img.shape)
    for i in range(pad_size,pad_size+height):
        for j in range(pad_size,pad_size+width):
            window = pad_img[i-pad_size:i+pad_size+1,j-pad_size:j+pad_size+1]
            g_r = (row_grad_mask*window).sum()
            g_c = (col_grad_mask*window).sum()
            if norm == 1:
                g = L1_norm(g_r,g_c)
            elif norm == 2:
                g = L2_norm(g_r,g_c)
            o = math.degrees(math.atan2(g_c,g_r))
            G[i,j]=g
            O[i,j]=o
    return G[pad_size:pad_size+height,pad_size:pad_size+width],O[pad_size:pad_size+height,pad_size:pad_size+width]

# def sobel_gradient(img,norm):
#     K=2
#     row_grad_mask =np.array([[-1,0,1],[-K,0,K],[-1,0,1]])/(K+2)
#     col_grad_mask =np.array([[1,K,1],[0,0,0],[-1,-K,-1]])/(K+2)
#     return gradient_magntitue_and_orientation(img,row_grad_mask,col_grad_mask,norm)

def convolution(img,kernel):
    height,width = img.shape
    kernel_size=kernel.shape[0]
    pad_size = kernel_size//2
    pad_img = np.pad(img,(pad_size,pad_size),'edge')
    G= np.zeros(pad_img.shape)
    for i in range(pad_size,pad_size+height):
        for j in range(pad_size,pad_size+width):
            window = pad_img[i-pad_size:i+pad_size+1,j-pad_size:j+pad_size+1]
            window=window[::-1,::-1]
            G[i,j]=(window*kernel).sum()
    return G[pad_size:pad_size+height,pad_size:pad_size+width]

# mode: L1-norm:"L1", L2-norm:"L2", entropy:"entropy", HOG:"HOG"
# For unused parameters, you can fill in any value. eg: no window size parameter for L1/L2-norm
def energy_function(img: np.ndarray,mode: string ,window_size:int):
    if mode == 'L1':
        G,_ = gradient_magntitue_and_orientation(img,2,1)
        return G
    elif mode == 'L2':
        G,_ = gradient_magntitue_and_orientation(img,2,2)
        return G
    elif mode == 'entropy':
        return entropy(img,window_size)
    elif mode == "HOG":
        return HOG(img,window_size)
    else:
        print(f"error:There is no option for this mode.")
        return
        
# for "vertical seam":seam_orient='v', for "horizontal seam":seam_orient='h'
def find_seam_1d(energy_map:np.ndarray, seam_num:int, seam_orient:string):
    height,width = energy_map.shape
    min_E = np.zeros((height,width))
    min_O = np.zeros((height,width))
    seam_map = np.zeros((height,width))
    if seam_orient == 'v':
        min_E[0][:]=energy_map[0][:]
        min_O[0][:]=-1
        for i in range(1,height):
            for j in range(0,width):
                e=energy_map[i][j]
                ancestor = {}
                if j == 0:
                    ancestor[1]=min_E[i-1][j]
                    ancestor[2]=min_E[i-1][j+1]
                elif j == width-1:
                    ancestor[0]=min_E[i-1][j-1]
                    ancestor[1]=min_E[i-1][j]
                else:
                    ancestor[0]=min_E[i-1][j-1]
                    ancestor[1]=min_E[i-1][j]
                    ancestor[2]=min_E[i-1][j+1]
                min_E[i][j]=min(ancestor.values())+e
                min_O[i][j]=min(ancestor,key=ancestor.get)
    elif seam_orient == 'h':
        min_E.transpose()[0][:]=energy_map.transpose()[0][:]
        min_O.transpose()[0][:]=-1
        for j in range(1,width):
            for i in range(0,height):
                e=energy_map[i][j]
                ancestor = {}
                if i == 0:
                    ancestor[1]=min_E[i][j-1]
                    ancestor[2]=min_E[i+1][j-1]
                elif i == height-1:
                    ancestor[0]=min_E[i-1][j-1]
                    ancestor[1]=min_E[i][j-1]
                else:
                    ancestor[0]=min_E[i-1][j-1]
                    ancestor[1]=min_E[i][j-1]
                    ancestor[2]=min_E[i+1][j-1]
                min_E[i][j]=min(ancestor.values())+e
                min_O[i][j]=min(ancestor,key=ancestor.get)
    smallest_k=np.array([(x,float('inf')) for x in range(seam_num)],dtype=[('index','<i4'),('total_energy','<f4')])
    if seam_orient == 'v':
        for j in range(width):
            for k in range(seam_num):
                if min_E[height-1][j] < smallest_k['total_energy'][k]:
                    smallest_k[k]=(j,min_E[height-1][j])
                    break
        for k in range(seam_num):
            j=smallest_k['index'][k]
            for i in reversed(range(height)):
                seam_map[i][j]=-1
                if min_O[i][j]==0:
                    j-=1
                elif min_O[i][j]==2:
                    j+=1
    elif seam_orient == 'h':
        for i in range(height):
            for k in range(seam_num):
                if min_E[i][width-1] < smallest_k['total_energy'][k]:
                    smallest_k[k]=(i,min_E[i][width-1])
                    break
        for k in range(seam_num):
            i=smallest_k['index'][k]
            for j in reversed(range(width)):
                seam_map[i][j]=-1
                if min_O[i][j]==0:
                    i-=1
                elif min_O[i][j]==2:
                    i+=1
    return seam_map