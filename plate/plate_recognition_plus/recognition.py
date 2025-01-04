# -*- coding: utf-8 -*-
import numpy as np
from PIL import ImageDraw, ImageFont, Image
import hyperlpr3 as lpr3
import cv2

# 1.定义数据结构：通过 `PlateInfo` 类来存储车牌识别的相关信息，包括车牌号、识别置信度、车牌框位置、车牌颜色、裁剪的车牌图像和最终结果图像。
class PlateInfo:
    def __init__(self):
        self.result_image = None  # 车牌框显示的图像
        self.box = None  # 车牌框的坐标
        self.color = None  # 车牌颜色
        self.confidence = None  # 车牌识别的置信度
        self.plate = None  # 车牌号
        self.crop_plate = None  # 裁剪出来的车牌图片


# - 调用 `hyperlpr3` 库进行车牌识别：使用 `LicensePlateCatcher` 类的实例 `catcher` 对输入图像进行车牌识别，获取识别结果 `results`。
class Recognition:
    def __init__(self, image):
        self.image = image  # 传入的图像
        self.plate_info = PlateInfo()  # 存储车牌信息的对象

    # 获取车牌信息
    def get_plate_info(self):
        # 使用hyperlpr3库的车牌识别功能
        catcher = lpr3.LicensePlateCatcher()
        results = catcher(self.image)  # 提取识别结果：从 `results` 中提取车牌号 `plate`、置信度 `confidence`、车牌框坐标 `box` 等信息。
        if results:
            # 提取识别结果的车牌号、置信度、车牌框坐标等
            plate, confidence, _, box = results[0]
            # 加载字体
            font_ch = ImageFont.truetype("platech.ttf", 20)
            # 绘制车牌框和车牌号：调用 `draw_plate_on_image` 方法，在原始图像上绘制车牌框和车牌号，生成结果图像 `result_image`。
            result_image = self.draw_plate_on_image(self.image, box, plate, font_ch)
            # 裁剪车牌图像：调用 `crop_plate` 方法，根据车牌框坐标 `box` 裁剪出车牌部分的图像 `crop_plate`
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

    #  绘制车牌框：使用 OpenCV 的 `cv2.rectangle` 函数，在图像上绘制红色的车牌框，框的坐标由 `box` 提供。
    #   - 绘制车牌号背景框：在车牌框上方绘制一个红色的矩形背景框，用于显示车牌号。
    #   - 绘制车牌号文本：将 OpenCV 图像转换为 PIL 图像，使用 PIL 的 `ImageDraw.Draw` 方法在背景框上绘制白色车牌号文本，字体由 `font_ch` 指定。
    #   - 转换回 OpenCV 图像：将绘制完文本的 PIL 图像转换回 NumPy 数组，以便后续处理和展示。
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

    #  将 NumPy 图像数组转换为 PIL 图像：使用 `Image.fromarray` 函数将输入的 NumPy 图像数组转换为 PIL 图像。
    #   - 根据车牌框坐标裁剪图像：使用 PIL 图像的 `crop` 方法，根据车牌框坐标 `box` 裁剪出车牌部分的图像。
    #   - 返回裁剪后的图像：将裁剪后的 PIL 图像转换回 NumPy 数组，并返回。
    def crop_plate(self, image, box):
        # 将NumPy图像数组转换为PIL图像
        img = Image.fromarray(image)
        # 根据车牌框坐标裁剪图像
        return np.array(img.crop(box))

    # - 定义颜色的 HSV 阈值范围**：根据蓝色、黄色和绿色在 HSV 色彩空间中的特征，定义它们的阈值范围。
    #   - **将车牌图像转换为 HSV 色彩空间**：使用 OpenCV 的 `cv2.cvtColor` 函数将车牌图像从 BGR 转换为 HSV 色彩空间。
    #   -生成颜色掩膜：使用 `cv2.inRange` 函数根据定义的 HSV 阈值范围生成蓝色、黄色和绿色的掩膜。
    #   - 统计白色像素*：分别统计每种颜色掩膜中白色区域（车牌的主要区域）的像素数。
    #   - 选择车牌颜色：比较三种颜色的白色像素数，选择数量最多的颜色作为车牌颜色，并返回。
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


