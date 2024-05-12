import argparse
import torch
import torch.nn as nn
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import create_dataloader_rgb_ir
from utils.torch_utils import select_device, time_synchronized
from utils.general import check_img_size, increment_path
from models.common import CrossViT

def test(data,
         weights,
         batch_size,
         imgsz,
         conf_thres,
         iou_thres,
         device='0',
         half_precision=True,
         opt=None):
    # 设定设备
    #set_logging()
    device = select_device(opt.device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 明确地加载模型，确保模型被加载到正确的设备上
    model = attempt_load(weights, map_location=device)  # 加载.pt权重文件
    for childs in model.children():
        if isinstance(childs, nn.Sequential):
            for child in childs.children():
                if isinstance(child, CrossViT):
                    print(type(child.fusion[0].attn))
                    child.fusion[0].attn.register_forward_hook(child.get_activation('crossVitOutput'))
                    print('GOT IT ***************')

    # 检查图像尺寸，确保与模型的最大步长兼容
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)
    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    model.to(device).eval()  # 将模型设置为评估模式并移到指定设备
    # 加载数据
    val_path_rgb = data['val_rgb']
    val_path_ir = data['val_ir']
    dataloader = create_dataloader_rgb_ir(val_path_rgb, val_path_ir, imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]
    s = ('%20s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.75', 'mAP@.5:.95')
    # 前向传播
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        img_rgb = img[:, :3, :, :]
        img_ir = img[:, 3:, :, :]

        with torch.no_grad():
            t = time_synchronized()
            out, train_out = model(img_rgb, img_ir, augment=augment)  # inference and training outputs
            if 'crossVitOutput' in ACTIVATION:
                intermediate_output = ACTIVATION['crossVitOutput']
                # 在这里可以进一步处理中间层输出，例如可视化、保存到文件等
            else:
                raise Exception("No intermediate output found in ACTIVATION.")
            t0 += time_synchronized() - t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='script.py')
    parser.add_argument('--weights', nargs='+', type=str, default='/home/watanabelab/multispectural-object-detection/liujiahao/runs/train/12.9/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='./data/multispectral/LLVIP.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=64, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    opt = parser.parse_args()

    test(opt.data,
         opt.weights,
         opt.batch_size,
         opt.img_size,
         opt.conf_thres,
         opt.iou_thres,
         opt=opt
         )
