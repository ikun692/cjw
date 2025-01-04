# -*- coding: utf-8 -*-
from tkinter import Tk, Frame, Label, Button, filedialog
from PIL import Image, ImageTk
import cv2
from recognition import Recognition


class MainGui:
    def __init__(self, master: Tk):
        self.master = master
        master.update()
        width, height = master.winfo_width(), master.winfo_height()

        # 定义左右两个Frame，增加边距和美观效果
        self.left_frame = Frame(master, width=width * (2 / 5), height=height, bg='white', bd=1, relief='solid')
        self.right_frame = Frame(master, width=width * (3 / 5), height=height, bg='white', bd=1, relief='solid')
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        # 定义图像显示Frame，设置固定尺寸和边框
        self.image_frame = Frame(self.left_frame, width=width * (2 / 5), height=250, bg='white', bd=1, relief='solid')
        self.image_frame2 = Frame(self.left_frame, width=width * (2 / 5), height=250, bg='white', bd=1, relief='solid')
        self.image_frame.grid(row=0, column=0, columnspan=2, pady=(10, 5))
        self.image_frame2.grid(row=4, column=0, columnspan=2, pady=(10, 5))

        # 显示文本标签，增加字体大小和对齐方式
        Label(self.left_frame, text="车牌", font=('Arial', 10, 'bold')).grid(row=5, column=0, pady=5, sticky='e')
        Label(self.left_frame, text="车牌号", font=('Arial', 10)).grid(row=6, column=0, pady=5, sticky='e')
        Label(self.left_frame, text="车牌颜色", font=('Arial', 10)).grid(row=7, column=0, pady=5, sticky='e')
        Label(self.left_frame, text="车牌类型", font=('Arial', 10)).grid(row=8, column=0, pady=5, sticky='e')
        Label(self.left_frame, text='图像预览', font=('Arial', 12, 'bold')).grid(row=1, column=0, columnspan=2, pady=10)

        # 按钮美化
        Button(self.left_frame, text="选择图片", command=self.choose_pic, width=20, bg='lightblue', font=('Arial', 10)).grid(row=2, column=0, columnspan=2, pady=10)
        Button(self.left_frame, text="开始识别", command=self.start_reco, width=20, bg='lightgreen', font=('Arial', 10)).grid(row=3, column=0, columnspan=2, pady=10)

        self.filenames = []
        self.image_width = int(width * (2 / 5))

        # 保持ImageTk.PhotoImage对象的引用
        self.imgtk_result = None
        self.imgtk_crop_plate = None

    def display_image(self):
        # 清空原有的图片，只删除预览图片区域的内容
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        # 如果有选中的图片，加载显示
        if self.filenames:
            img = Image.open(self.filenames[0]).resize((self.image_width, 250))
            self.imgtk_result = ImageTk.PhotoImage(image=img)

            # 确保保持图像引用，防止图片消失
            Label(self.image_frame, image=self.imgtk_result).pack(side='left', expand=True)

    def choose_pic(self):
        # 选择图片
        self.filenames = filedialog.askopenfilenames()
        self.display_image()

    def start_reco(self):
        # 开始识别
        if self.filenames:
            image = cv2.imread(self.filenames[0])
            rg = Recognition(image)
            result = rg.get_plate_info()
            self.show_plate_info(result)

    def show_plate_info(self, result):
        # 清空检测结果区域
        for widget in self.image_frame2.winfo_children():
            widget.destroy()

        # 处理并显示识别后的图像
        if result.result_image is not None:
            result_img = cv2.cvtColor(result.result_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(result_img).resize((self.image_width, 250))
            self.imgtk_result = ImageTk.PhotoImage(image=img)
            Label(self.image_frame2, image=self.imgtk_result).grid(row=0, column=0)

        # 处理并显示裁剪后的车牌图像
        if result.crop_plate is not None:
            crop_plate = cv2.cvtColor(result.crop_plate, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(crop_plate).resize((100, 25))
            self.imgtk_crop_plate = ImageTk.PhotoImage(image=img)
            Label(self.left_frame, image=self.imgtk_crop_plate).grid(row=5, column=1, pady=5)

        # 显示车牌信息
        Label(self.left_frame, text=result.plate, font=('Arial', 10, 'bold')).grid(row=6, column=1, pady=5)
        Label(self.left_frame, text=result.color, font=('Arial', 10)).grid(row=7, column=1, pady=5)
        Label(self.left_frame, text='新能源车' if result.color == '绿色' else '普通车', font=('Arial', 10)).grid(row=8, column=1, pady=5)



