
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model

folder_path = 'eyes/'
classes = ['an','n']
num_classes = len(classes)

image_w = 68
image_h = 24

X = []
Y = []

img_data = []

for index, cla in enumerate(classes):
    label = [0 for i in range(num_classes)]
    label[index] = 1
    image_dir = folder_path + cla + '/'
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            img = cv2.imread(image_dir + filename)
            img = cv2.resize(img, None, fx = image_w/img.shape[1], fy = image_h / img.shape[0])
            img_data.append(img)
            X.append(img/256)
            Y.append(label)


X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
model = Sequential()
model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
def testcut(path):
    image_w = 68
    image_h = 24
    img = cv2.imread(path)
    img = cv2.resize(img, None, fx = image_w/img.shape[1], fy = image_h/img.shape[0])
    return img/256




src = []
name = []
test = []
image_dir = 'test5/'
for file in os.listdir(image_dir):
    src.append(image_dir + file)
    name.append(file)
    test.append(testcut(image_dir+file))

test = np.array(test)
history = model.fit(X, Y, batch_size=32, nb_epoch=2)

print(history)



predict = model.predict_classes(test)
count_num1 = 0
count_num2 = 0
for i in range(len(test)):
    print(name[i] + 'predict : ' + str(classes[predict[i]]))
    if predict[i] == 0:
        count_num1 += 1
    else:
        count_num2 += 1
total_count = count_num1 + count_num2
#동영상 길이는 10초,


print(count_num1 / total_count)
if count_num1 / total_count > 0.1 and count_num1 / total_count < 0.5:
    print("환자의 가능성이 있습니다.")
    print("눈동자가 튀는 현상이 적게 발견되었습니다.")
    print(count_num1*1.5 / total_count * 100," 확률로 환자입니다.")
elif count_num1 / total_count >= 0.5 and count_num1 / total_count <0.66:
    print("환자의 가능성이 있습니다.")
    print("눈동자가 튀는 현상이 적지않게 발견되었습니다. ")
    print(count_num1 * 1.5 / total_count * 100," 확률로 환자입니다.")
elif count_num1 / total_count >= 0.66:
    print("환자의 가능성이 있습니다.")
    print("눈동자가 튀는 현상이 많이 발견 되었습니다.")
    print(count_num1 / total_count)
else:
    print("정상인 입니다.")
