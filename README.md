# DIP_term_project

- 摘要：Seam carving 是用來調整圖片大小的演算法，其特色在於調整大小的同時會考慮到圖片內容：透過調整比較不重要的像素，使得圖片不會有明顯的失真問題。而像素的重要性取決於「能量函數」。


### 能量函數
energy_function.py
- 計算像素的能量值。
- 我們實作了 Entropy, L1-norm of gradient, L2-norm of gradient, HOG, forward entropy


### 找到 Seam
1 find_seam.py
- 找到單一方向：「水平」或是「垂直」的 Seam

2. find_seam_2d_dp.py
- 尋找可「曲折」的 Seam，同作者實作更改二維大小的方法

### Seam carving 演算法：調整圖片大小
1. insert_seam.py
- 實作插入 seam

2.1 remove_seam.py---------------------> 事實上有了 2.2 就不用它了
- 實作移除 seam

2.2 object_preservation.py
- 加上保護遮罩以後執行 seam removal

3. seam_carve.py
- 給定原圖與目標圖片大小以實作 Seam carving 演算法
- 使用 insert_seam.py, remove_seam.py

### 放大圖片主體
content_amplification.py
- 放大圖片主體

### 移除特定物體
1.1 object_removal.py ------------------------> 事實上有了 1.2 就不用它了
- 在螢幕上點擊保護遮罩，接著進行移除特定物體
- seam insertion 是直接套用 insert_seam

1.2 object_removal_with_protective_mask.py
- 移除遮罩 + 保護遮罩
- 改進 seam insertion 方法：加入在剛才移除的地方

### 其他調整大小的方法
other_resize.py
- 直接裁切、移除欄/列、移除各欄/列的不重要元素

### 顯示 seam carving 過程
show.py



