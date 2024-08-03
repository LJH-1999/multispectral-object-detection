import cv2
import os

# 标签映射表
label_map = {
    0: 'person',
    1: 'bicycle',
    2: 'car'
    # 添加更多标签根据需要
}

# 读取图像
#/home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/images/visible/val/image.jpg
#/home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/images/infrared/val/image.jpg
image_path = '/home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/images/infrared/val/240260.jpg'
image = cv2.imread(image_path)

# 从文件中读取所有标签数据（假设每行一个目标）
# /home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/labels/visible/val/label.txt
#/home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/labels/infrared/val/label.txt
label_file = '/home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/labels/infrared/val/240260.txt'

# 读取所有目标的标签数据
with open(label_file, 'r') as f:
    lines = f.readlines()

# 处理每个目标的标签数据
for line in lines:
    parts = line.strip().split()
    label_index = int(parts[0])  # 获取标签索引
    label = label_map[label_index]  # 根据索引获取标签名
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
    color = (255, 0, 0)  # 蓝色
    thickness = 2
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    # 在边界框上方绘制标签
    label_text = label  # 使用标签名
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    label_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
    label_origin = (x_min, y_min - 10)  # 标签位置
    label_background_top_left = (x_min, y_min - label_size[1] - 10)
    label_background_bottom_right = (x_min + label_size[0], y_min)

    # 绘制标签背景
    cv2.rectangle(image, label_background_top_left, label_background_bottom_right, color, -1)

    # 绘制标签文字
    cv2.putText(image, label_text, label_origin, font, font_scale, (255, 255, 255), font_thickness)

# 获取输入图像的文件名和扩展名
image_name = os.path.basename(image_path)
image_name_without_extension, extension = os.path.splitext(image_name)

# 构造输出图像的文件路径和名称
output_image_name = f'inference_{image_name_without_extension}{extension}'
output_path = os.path.join('/home/watanabelab/multispectural-object-detection/liujiahao/LLVIP_2/printbbox/val/ir/', output_image_name)

# 保存带有边界框的图像到指定路径
cv2.imwrite(output_path, image)