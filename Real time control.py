import cv2
from cv2 import dnn
import numpy as np

cap = cv2.VideoCapture(0)
print(cv2.__version__)
class_name = ['0', '1', '2',  '4', '5',]
net = dnn.readNetFromTensorflow(r'F:\frozen_graph.pb')

# 移除视频数据的背景噪声
def _remove_background(frame):
    fgbg = cv2.createBackgroundSubtractorMOG2()  # 利用BackgroundSubtractorMOG2算法消除背景
    # fgmask = bgModel.apply(frame)
    fgmask = fgbg.apply(frame)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# 视频数据的人体皮肤检测
def _bodyskin_detetc(frame):
    # 肤色检测: YCrCb之Cr分量 + OTSU二值化
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  # 分解为YUV图像,得到CR分量
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)  # 高斯滤波
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # OTSU图像二值化
    return skin

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        _, frame = cap.read() #获取帧

        src_image = frame #帧图像
        cv2.rectangle(src_image, (300, 100), (600, 400), (0, 255, 0), 1, 4)#选择框

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #将BGR格式转换成RGB格式
        pic = frame[100:400, 300:600]#ROI
        pic = _remove_background(pic)
        pic = _bodyskin_detetc(pic)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        cv2.imshow("pic2", pic)

        pic = cv2.resize(pic, (100, 100))

        blob = cv2.dnn.blobFromImage(pic,
                                     scalefactor=1.0 / 225.,
                                     size=(100, 100),
                                     mean=(0, 0, 0),
                                     swapRB=False,
                                     crop=False)
        # blob = np.transpose(blob, (0,2,3,1))
        net.setInput(blob)
        out = net.forward()
        out = out.flatten()

        classId = np.argmax(out)
        print("classId",classId)
        print("预测结果为：", class_name[classId])
        cv2.putText(src_image, str(class_name[classId]), (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, 4)


        cv2.imshow("pic", src_image)
        if cv2.waitKey(10) == 27:
            break
    cap.release()


