# DIP_term_project

- 摘要：Seam carving 是用來調整圖片大小的演算法，其特色在於調整大小的同時會考慮到圖片內容：透過調整比較不重要的像素，使得圖片不會有明顯的失真問題。而像素的重要性取決於「能量函數」。


#### 能量函數
energy_function.py
- 計算像素的能量值。
- 我們實作了 Entropy, L1-norm of gradient, L2-norm of gradient, HOG, forward entropy


#### 找到 Seam
find_seam.py
- 找到單一方向：「水平」或是「垂直」的 Seam

find_seam_2d_dp.py
- 尋找可「曲折」的 Seam，同作者實作更改二維大小的方法


