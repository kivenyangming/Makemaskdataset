import os
import cv2
import uuid

import numpy
from PIL import Image


def format_image(image):
    # image如果为彩色图：image.shape[0][1][2](水平、垂直像素、通道数)
    if len(image.shape) > 2 and image.shape[2] == 3:
        # 将图片变为灰度图
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示）
        # 调整scaleFactor参数的大小，可以增加识别的灵敏度，推荐1.1
        #CASC_PATH = 你的haarcascade_frontalface_alt2.xml文件地址
        CASC_PATH = './haarcascade_frontalface_alt2.xml' #
        cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
        faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    # 如果图片中没有检测到人脸，则返回None
    if not len(faces) > 0:
        return None, None
    # max_are_face包含了人脸的坐标，大小
    max_are_face = faces[0]
    # 在所有人脸中选一张最大的脸
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face

    # 这两步可有可无
    face_coor = max_are_face
    image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    # 调整图片大小，变为48*48
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("problem during resize")
        return None, None

    return image, face_coor


if __name__ == "__main__":
    NewPersonImgPath = 'C:/Users/kiven/Desktop/do_img/score/images/val/'  #
    NewPersonImgPathLine = os.listdir(NewPersonImgPath)  # 获取工程车辆文件夹
    for NewPersonName in NewPersonImgPathLine:
        NewPersonImage = Image.open(NewPersonImgPath + NewPersonName)
        frame = cv2.cvtColor(numpy.asarray(NewPersonImage), cv2.COLOR_RGB2BGR)
        (p_image, face_coor) = format_image(frame)
        if face_coor is not None:
            # 获取人脸的坐标,并用矩形框出
            [x, y, w, h] = face_coor
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            x0 = x
            y0 = y
            x1 = x + w
            y1 = y + h
            crop_img = frame[y0:y1, x0:x1]  # x0,y0为裁剪区域左上坐标；x1,y1为裁剪区域右下坐标
            crop_imgs = cv2.resize(crop_img, (100, 100), interpolation=cv2.INTER_CUBIC)
            img_name = uuid.uuid1()
            cv2.imwrite('C:/Users/kiven/Desktop/do_img/faces/%s.jpg' % img_name, crop_imgs)  # save_path为保存路径




