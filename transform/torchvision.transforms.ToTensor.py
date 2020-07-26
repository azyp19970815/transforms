import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt


img_path = "1.png"

# transforms.ToTensor()
transform1 = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
)

##numpy.ndarray
img = cv2.imread(img_path)  # 读取图像
plt.imshow(img)
plt.show()

b, g, r = cv2.split(img)
img = cv2.merge([r, g, b])  # cv2.imread读取图像格式为b,g,r,但是plt显示按照rgb次序
plt.imshow(img)
plt.show()

img1 = transform1(img)  # 归一化到 [0.0,1.0]
print("img.shape = ", img.shape)
print("img1.shape = ", img1.shape)

print("img = ", img)
print("img1 = ", img1)

# 转化为numpy.ndarray并显示
img_1 = img1.numpy()*255
print(img_1)

# 转化为uint8类型（8位无符号整型）
img_1 = img_1.astype('uint8')
print(img_1)

img_1 = np.transpose(img_1, (1, 2, 0))
print(img_1)
plt.imshow(img_1)
plt.show()
print("\n-----------\n")
org = plt.imread("1.png")
plt.imshow(org)
plt.show()
