import os
import cv2

path1 = "D:\\originalPics\\2002\\"
path2 = "D:\\originalPics\\2003\\"
Res_path = "D:\\FaceDataset\\"

list_path1_subDir = os.listdir(path1)
list_path2_subDir = os.listdir(path2)
SubPath = []

for con in list_path1_subDir:
    SubPath = os.listdir(path1 + con + "\\")
    for index in range(len(SubPath)):
        imgPath = path1 + con + "\\" + SubPath[index] + "\\big\\"
        imgList = os.listdir(imgPath)
        for j in range(len(imgList)):
            img_path = path1 + con + "\\" + SubPath[index] + "\\big\\" + imgList[j]
            img = cv2.imread(img_path)
            cv2.imwrite(Res_path + imgList[j] + ".jpg", img)
