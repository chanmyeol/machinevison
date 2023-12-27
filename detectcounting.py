import csv

import cv2
import numpy as np
import os
import pandas as pd
import time


# 读取图像并转换为 HSV 颜色空间
def counting(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 设定红色和绿色的颜色范围
    lower_red1 = np.array([0, 43, 15])
    upper_red1 = np.array([34, 255, 255])
    lower_red2 = np.array([156, 43, 15])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([78, 43, 15])
    upper_green = np.array([90, 255, 255])

    # 限制图像在红色和绿色范围内
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_green, upper_green)

    # 合并两个 mask
    mask = mask1 + mask2 + mask3

    # 去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # (5,5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # 找到苹果的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    countapple = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        k = cv2.isContourConvex(contour)
        if area > 200:
            countapple = countapple + 1
        x, y, w, h = cv2.boundingRect(contour)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return countapple


# 显示结果图像
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# print(countapple)
mainFolder = "/Users/chengzi/Downloads/counting/train/images"
array_of_img = []


def read_directory(directory_name):
    for filename in os.listdir(directory_name):
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)


read_directory(mainFolder)

# data = pd.read_csv("val_ground_truth.csv")
# import csv

# 输入 CSV 文件路径
csv_file_path = 'train_ground_truth.csv'

# 创建一个空字典
data = {}

# 打开 CSV 文件进行读取
with open(csv_file_path, 'r') as csv_file:
    # 创建 CSV 读取器
    csv_reader = csv.reader(csv_file)

    # 遍历 CSV 文件的每一行
    for row in csv_reader:
        # 确保行中至少有两个值
        if len(row) >= 2:
            # 使用第一个值作为键，第二个值作为值
            key = row[0]
            value = row[1]

            # 将键值对添加到字典中
            data[key] = value

# print("生成的字典:")
# print(data_dict["images_02473.png"])
# print(len(data_dict))

# print(data['count'])
# with open("val_ground_truth.csv", 'r') as f:
#     reader = csv.reader(f)
#     result = list(reader)
# y = result[1]
# y = list(map(int,y))
# print(y)
# with open("ground_truth.csv", 'r') as f:
#     reader = csv.reader(f)
#     result = list(reader)
# y = result[1]
# y = list(map(int,y))
# print(y)
# y = data['count'].tolist()
# print(y)

returntrue = 0
# print(filename)
# for file in
# start_time = time.time()
# for i in range(0,len(array_of_img)):
#     if counting(array_of_img[i]) ==y[i]:
#         returntrue = returntrue +1
#
# rate = returntrue/len(array_of_img)
# end_time = time.time()
# print(rate)
# print('cost %f second' % (end_time - start_time))
truerate = 0
thres = 10
p = 0
start_time = time.time()

for filename in os.listdir(mainFolder):
    # print(filename)
    # print(mainFolder+ "/"+filename)
    # print(data[filename])
    # print(type(data[filename]))

    m = 0
    img = cv2.imread(mainFolder+ "/"+filename)
    print("count:",counting(img),"answer:",int(data[filename]))
    if((counting(img) == int(data[filename]))):
        m = 1
    else:
        m = 0

    p = p + 1
    # print(m)
    truerate = truerate + m
    # k = (((1 - (abs(counting(img) - int(data[filename])) / int(data[filename]))) < 1) & (
    #             (1 - (abs(counting(img) -int( data[filename])) / int(data[filename]))) > 0))
    # if k:
    #     m = 1 - (abs(counting(img) - int(data[filename])) / int(data[filename]))
    #     p = p + 1
    #     print(m)
    #     truerate = truerate + m
end_time = time.time()
print(p)
t = truerate / p

#print('get %f bad' % (p-0))
print('get %f acc' % (t-0))
print('cost %f second' % (end_time - start_time))
