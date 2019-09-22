# eyes-shiver-predict
## 1. eyes-shiver-predict?

1. 눈동자의 움직임을 통해 눈 떨림의 정도나 본인이 눈떨림 증상 환자임을 유추 할 수 있는 프로그램
1. 눈움직임의 동영상을 촬영 후 이미지를 쪼개어 데이터를 수집
1. 정면을 주시하는 눈동자 이미지 데이터와 정면에서 벗어난 눈동자 이미지 데이터를 CNN을 통해 학습시켜 눈동자의 벗어남을 측정

![eye](https://user-images.githubusercontent.com/50629716/65382429-d9e74000-dd3f-11e9-9433-029e429dbdcd.png)

## 2. 환경 설정 및 라이브러리

1. Python 3.6v (Conda는 가상환경 설정), 플랫폼은 Pycharm을 이용했음.
1. Tensorflow
1. Opencv-Contrib
1. keras
1. Numpy
1. Pillow
