import cv2
import os
import numpy as np

# 读取图像
#/home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/images/visible/val/image.jpg
image_path = '/home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/images/infrared/val/240260.jpg'
image = cv2.imread(image_path)

# 从文件中读取标签数据
# /home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/labels/visible/val/label.txt
label_file = '/home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/labels/infrared/val/240260.txt'

# label.txt中的数据格式
# 0 0.2890625 0.89453125 0.084375 0.2109375
with open(label_file, 'r') as f:
    line = f.readline().strip()
    parts = line.split()
    label = int(parts[0])
    bbox = tuple(map(float, parts[1:]))

# 转换相对坐标为绝对坐标
height, width = image.shape[:2]
x = int(bbox[0] * width)
y = int(bbox[1] * height)
w = int(bbox[2] * width)
h = int(bbox[3] * height)

# 计算边界框的左上角和右下角坐标
x_min = x - int(w / 2)
y_min = y - int(h / 2)
x_max = x + int(w / 2)
y_max = y + int(h / 2)

# 绘制边界框
color = (0, 255, 0)  # 绿色
thickness = 2
image_with_box = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

# 获取输入图像的文件名和扩展名
image_name = os.path.basename(image_path)
image_name_without_extension, extension = os.path.splitext(image_name)

# 构造输出图像的文件路径和名称
output_image_name = f'output_{image_name_without_extension}{extension}'
#/home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/printbbox/val/rgb/
output_path = os.path.join('/home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/printbbox/val/ir/', output_image_name)

# 保存带有边界框的图像到指定路径
cv2.imwrite(output_path, image_with_box)

# 显示结果（可选）
cv2.imshow('Image with Bounding Box', image_with_box)
cv2.waitKey(0)
cv2.destroyAllWindows()