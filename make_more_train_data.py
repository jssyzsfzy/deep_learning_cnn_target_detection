import os
import cv2
import numpy as np
import imutils
# D:\Desktop\yolo\out\more_img/
path = r'D:\py_project\tor_my\save_camera'
save_path = r'D:\Desktop\yolo\out\more_img/'
path_list = os.listdir(path)
for i in path_list:
    img_path = path+'/'+i
    img = cv2.imread(img_path)
    # img_re = cv2.flip(img, 1)
    # kernel = np.ones((4, 4), np.uint8)
    # 图像开运算
    # result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # img_gs = cv2.GaussianBlur(img, (7, 7), 0)
    rot1 = imutils.rotate_bound(img, 15)
    rot2 = imutils.rotate_bound(img, -15)
    rot3 = imutils.rotate_bound(img, 45)
    rot4 = imutils.rotate_bound(img, -45)
    img1 = cv2.resize(rot1, (640, 640))
    img2 = cv2.resize(rot2, (640, 640))
    img3 = cv2.resize(rot3, (640, 640))
    img4 = cv2.resize(rot4, (640, 640))
    # cv2.imwrite(save_path + i.split('.')[0] + 'fl.jpg', img_re)
    cv2.imwrite(save_path + i.split('.')[0] + 'r15.jpg', img1)
    # cv2.imwrite(save_path + 'kai_' + i, result)
    # cv2.imwrite(save_path + 'gs_' + i, img_gs)
    cv2.imwrite(save_path + i.split('.')[0] + 'r165.jpg', img2)
    cv2.imwrite(save_path + i.split('.')[0] + 'r45.jpg', img3)
    cv2.imwrite(save_path + i.split('.')[0] + 'r145.jpg', img4)
    print(i + ' have saved more img')
    # cv2.waitKey(0)

