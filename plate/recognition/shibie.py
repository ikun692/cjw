# -*- coding: utf-8 -*-
import numpy as np
from PIL import ImageDraw, ImageFont, Image
import hyperlpr3 as lpr3
import cv2


# 定义PlateInfo类，存储车牌信息
class PlateInfo:
    def __init__(self):
        self.result_image = None  # 车牌框显示的图像
        self.box = None  # 车牌框的坐标
        self.color = None  # 车牌颜色
        self.confidence = None  # 车牌识别的置信度
        self.plate = None  # 车牌号
        self.crop_plate = None  # 裁剪出来的车牌图片


# 定义Recognition类，用于车牌识别
class Recognition:
    def __init__(self, image):
        self.image = image  # 传入的图像
        self.plate_info = PlateInfo()  # 存储车牌信息的对象

    # 获取车牌信息
    def get_plate_info(self):
        # 使用hyperlpr3库的车牌识别功能
        catcher = lpr3.LicensePlateCatcher()
        results = catcher(self.image)  # 获取识别结果

        if results:
            # 提取识别结果的车牌号、置信度、车牌框坐标等
            plate, confidence, _, box = results[0]
            # 加载字体
            font_ch = ImageFont.truetype("platech.ttf", 20)
            # 在图像上画出车牌框和车牌号
            result_image = self.draw_plate_on_image(self.image, box, plate, font_ch)
            # 裁剪出车牌部分图像
            crop_plate = self.crop_plate(self.image.copy(), box)
            # 获取车牌颜色
            color = self.get_plate_color(crop_plate.copy())

            # 将识别到的信息存入PlateInfo对象中
            self.plate_info.plate = plate
            self.plate_info.result_image = result_image
            self.plate_info.box = box
            self.plate_info.confidence = confidence
            self.plate_info.color = color
            self.plate_info.crop_plate = crop_plate

        return self.plate_info

    # 在图像上画出车牌框并显示车牌号
    def draw_plate_on_image(self, img, box, text, font):
        # 获取车牌框的坐标
        x1, y1, x2, y2 = box
        # 绘制车牌框，红色，线宽为2
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # 绘制车牌号背景框，红色，矩形高度为30（用负值设置填充）
        cv2.rectangle(img, (x1, y1 - 30), (x2, y1), (0, 0, 255), -1)
        # 将OpenCV图像转换为PIL图像，以便使用PIL绘制文本
        data = Image.fromarray(img)
        draw = ImageDraw.Draw(data)
        # 在车牌框上绘制车牌号
        draw.text((x1 + 1, y1 - 30), text, fill=(255, 255, 255), font=font)
        # 将PIL图像转换回NumPy数组
        return np.asarray(data)

    # 根据车牌框裁剪车牌部分图像
    def crop_plate(self, image, box):
        # 将NumPy图像数组转换为PIL图像
        img = Image.fromarray(image)
        # 根据车牌框坐标裁剪图像
        return np.array(img.crop(box))

    # 获取车牌颜色
    def get_plate_color(self, plate_image):
        # 定义蓝色、黄色和绿色的HSV阈值范围
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([0, 3, 116])
        upper_green = np.array([76, 211, 255])

        # 将车牌图像从BGR转换为HSV色彩空间
        hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
        # 生成三个颜色的掩膜，分别代表蓝色、黄色和绿色
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # 统计每种颜色掩膜中白色区域（车牌的主要区域）的像素数
        blue_white = np.sum(mask_blue == 255)
        yellow_white = np.sum(mask_yellow == 255)
        green_white = np.sum(mask_green == 255)

        # 选择白色像素最多的颜色作为车牌颜色
        colors = ['蓝色', '黄色', '绿色']
        color = colors[np.argmax([blue_white, yellow_white, green_white])]
        print('车牌的颜色为:', color)
        return color

# PlateInfo 类：
#
# PlateInfo 类用来存储车牌识别的相关信息，包括车牌号、识别置信度、车牌框位置、车牌颜色、裁剪的车牌图像和最终结果图像。
# Recognition 类：
#
# Recognition 类接收图像作为输入，并通过多个方法实现车牌的识别与处理。主要功能包括获取车牌信息、绘制车牌框、裁剪车牌图像和判断车牌颜色。
# get_plate_info 方法：
#
# 这是核心方法，调用 hyperlpr3 进行车牌识别，提取车牌号、置信度、车牌框位置等信息，并生成相应的结果图像、裁剪的车牌图像和车牌颜色。
# draw_plate_on_image 方法：
#
# 该方法在图像上绘制车牌框和车牌号，使用 OpenCV 绘制车牌框和背景，然后使用 PIL 来绘制文本，最后返回绘制后的图像。
# crop_plate 方法：
#
# 该方法根据车牌框坐标裁剪图像，返回车牌部分的图像。
# get_plate_color 方法：
#
# 该方法根据裁剪后的车牌图像，使用 HSV 色彩空间和掩膜技术判断车牌的颜色。通过统计每种颜色的白色像素点数，选择数量最多的颜色作为车牌颜色。
