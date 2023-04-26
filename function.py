import cv2
import numpy as np
from numpy import unravel_index
from matplotlib import pyplot as plt
import math

# energy function
# gray_scale
def L1_norm(g_r,g_c):
    return abs(g_r)+abs(g_c)

def L2_norm(g_r,g_c):
    return math.hypot(g_r,g_c)

def entropy(img,win_size):
    height,width=img.shape
    pad_size = win_size//2
    pad_img = np.pad(img,(pad_size,pad_size),'edge')
    E=np.zeros(pad_img.shape)
    for i in range(pad_size,pad_size+height):
        for j in range(pad_size,pad_size+width):
            window = pad_img[i-pad_size:i+pad_size+1,j-pad_size:j+pad_size+1]
            hist,bins = np.histogram(window.ravel(),256,[0,256])
            pdf = hist/window.size
            pdf=-pdf*math.log2(pdf)
            e=pdf.sum()
            E[i][j]=e
    return E[pad_size:pad_size+height,pad_size:pad_size+width]

def HOG(img):
    height,width=img.shape
    row_grad_mask=np.array([[0,0,0],[-1,0,1],[0,0,0]])
    col_grad_mask =np.array([[0,1,0],[0,0,0],[0,-1,0]])
    kernel_size=3
    pad_size = kernel_size//2
    # pad_img = cv2.copyMakeBorder(img,pad_size,pad_size,pad_size,pad_size,cv2.BORDER_REFLECT)
    pad_img = np.pad(img,(pad_size,pad_size),'edge')
    G= np.zeros(pad_img.shape)
    O= np.zeros(pad_img.shape)
    for i in range(pad_size,pad_size+height):
        for j in range(pad_size,pad_size+width):
            window = pad_img[i-pad_size:i+pad_size+1,j-pad_size:j+pad_size+1]
            g_r = abs((row_grad_mask*window).sum())
            g_c = abs((col_grad_mask*window).sum())
            g=L2_norm(g_r,g_c)
            o = math.degrees(math.atan2(g_c,g_r))
            G[i,j]=g
            O[i,j]=o
    G=G[pad_size:pad_size+height,pad_size:pad_size+width]
    O=O[pad_size:pad_size+height,pad_size:pad_size+width]
    for i in range(0,height):
        for j in range(0,width):
            if 0<=O[i][j]<=20:
            elif 
            elif
            elif
            elif
            elif
            elif
            elif
            elif
            elif
            elif
    return

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