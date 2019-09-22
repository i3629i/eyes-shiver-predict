import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_casecade = cv2.CascadeClassifier('haarcascade_eye.xml')

# def find_eyes(image_data):
#     img = cv2.imread(image_data)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     eyes = eye_casecade.detectMultiScale(gray,1.3,5)
#     for (x,y,w,h) in eyes:
#         cv2.rectangle(img, (x,y), (x+w, y+h), (255,155,0),2)
#         cropped = img[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
# #         roi_gray = gray[y:y+h, x:x+w]
# #         roi_color = img[y:y+h, x:x+w]
# #         eyes = eye_casecade.detectMultiScale(roi_gray)
# #     cv2.imshow('Image view', img)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()

# src = cv2.imread('frame1.jpg',cv2.IMREAD_GRAYSCALE)
# dst = src.copy()
# dst = src[0:700, 800:1500]
# cv2.imshow('src',src)
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()
# width = 500
# height = 500
# bpp = 3
#
# img = cv2.imread('grayframe0.jpg')
# height, width = img.shape[:2]
#
# img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# lower_black = (10-10,0,0)
# upper_black = (10+10,90,90)
# img_mask = cv2.inRange(img_hsv,lower_black,upper_black)
#
# img_result = cv2.bitwise_and(img,img,mask= img_mask)
# cv2.imshow('img_color',img)
# cv2.imshow('img',img_mask)
# cv2.imshow('img_re',img_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # (250,250)이 중심인 반지름 10인 파란색으로 채워진 원을 그립니다.
# cv2.circle(img, (250, 250), 10, (255, 0, 0), -1)
# # (250,250)이 중심인 반지름이 100인 선굵기가 1인 빨간색 원을 그립니다.
# cv2.circle(img, (250, 250), 100, (0, 0, 255), 1)
#
# cv2.imshow("result", img)
# cv2.waitKey(0);



#
# vidcap = cv2.VideoCapture('a1.mp4')
# success,image = vidcap.read()
#
# cut = cv2.selectROI('이미지자르기',image,fromCenter=True,showCrosshair=True)
#
# count = 0
# success = True
# while success:
#   success,image = vidcap.read()
#   if not success:
#     exit()
#   image = image[cut[1]:(cut[3] + cut[1]), cut[0]:(cut[0] + cut[2])]
#   img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   cv2.imwrite("frame%d.jpg" % count, img)     # save frame as JPEG file
#   if cv2.waitKey(10) == 27:                     # exit if Escape is hit
#       break
#   count += 1





video_path = 'n6.mp4'
cap = cv2.VideoCapture(video_path)

OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}

if not cap.isOpened():
    exit()

tracker = cv2.TrackerCSRT_create()

ret, img = cap.read()
cv2.namedWindow('Window')
cv2.imshow('Window', img)

#setting ROI ROi를 설정해서 rect로 변환

x = int(img.shape[0] / 10)
y = int(img.shape[1] / 10)
print(img.shape[0],img.shape[1])
print(x,y)

out = cv2.selectROI('Window',img,fromCenter=True,showCrosshair=True)
img = img[out[1]:(out[3]+out[1]),out[0]:(out[0]+out[2])]
# img = img[(out[1]-y):(out[1]+y),(out[0]-x):(out[0]+x)]
img = cv2.resize(img,dsize=(680,240),interpolation=cv2.INTER_AREA)
cir = cv2.selectROI('Window',img, fromCenter=True, showCrosshair=True)
print(cir)
cv2.destroyWindow('Window')
# tracker 초기화
tracker.init(img, cir)

list_left = []
list_top = []

list_x=[]
list_y=[]
result_X = []
result_y = []
i = -1
j = -1
count = 0
while True:
    ret, img = cap.read()
    if not ret:
        exit()
    img = img[out[1]:(out[3] + out[1]), out[0]:(out[0] + out[2])]
    img = cv2.resize(img, dsize=(680, 240), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    success, box = tracker.update(img) # success 는 boolean , box는 데이터 값
    left, top, w, h =[int(v) for v in box]
    # cv2.circle(img,(int((left+w/2)),int((top+h/2))),int(w/2),(255,255,0),2)
    x = left + w / 2
    # if x < 300 or x > 360:
    #   cv2.imwrite("4.1 %d.jpg" % count, img)
    cv2.imwrite("n5.mp4 %d.jpg" % count, img)
    count += 1
    #cv2.rectangle(img, pt1=(left, top),pt2 =(left+w, top+h),color = (255,255,0),thickness=2)
    print(left+w / 2)
    # print("(x,y):",left+w/2,top+h/2) #중심의 좌표값


    list_left.append(left)
    list_top.append(top)
    list_x.append(left + w / 2)
    list_y.append(top + h / 2)
    cv2.imshow('window', img)
    # print("(x,y):",left+w/2,top+h/2) #중심의 좌표값
    # i = i + 1
    # j = j + 1
    # if i > 0 and i < len(list_x):
    #     result_X.append(list_x[i] - list_x[i - 1])
    # if j > 0 and j < len(list_y):
    #     result_y.append(list_y[i] - list_y[i - 1])

    data = {"left좌표":list_left, "top좌표":list_top, "x좌표": list_x, "y_좌표": list_y}
    df = pd.DataFrame(data)
    # df_min_max = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    # df.to_csv('이석증4.2.csv', header=True)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break#그냥 꺼져 버릴 수 있음
cap.release()
cv2.destroyWindow()