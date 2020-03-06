import cv2
import numpy as np
from numpy import linalg as lg

def funcPCA():
    v_path = r"D:\Programming\MATLAB\video_prog\MVI_2883.MOV"
    vid = cv2.VideoCapture(v_path)
    flag = vid.isOpened()
    if flag:
        print("打开摄像头成功")
    else:
        print("打开摄像头失败")
    ret, frame = vid.read()
    size = (np.int(1080 * 0.2), np.int(720 * 0.2))
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
    SampleFrameNum = 250
    chn = 1
    FrameSamples = np.zeros(shape=(frame.shape[0] * frame.shape[1], SampleFrameNum), dtype=np.uint8)
    cv2.namedWindow("Frames Sample", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Frames Sample with PCA", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("rePic", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Sub", cv2.WINDOW_KEEPRATIO)
    cv2.namedWindow("Current frame", cv2.WINDOW_KEEPRATIO)
    NumFlag = 0
    # pca = 184000
    pca = 40000
    flag = False
    while 1:
        NumFlag += 1
        ret, frame = vid.read()
        if ret == 0:
            break
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        FrameInCol = frame.reshape((frame.shape[0] * frame.shape[1]))
        temp = FrameSamples[:, 1:SampleFrameNum]
        FrameSamples[:, 0:SampleFrameNum - 1] = temp
        FrameSamples[:, SampleFrameNum - 1] = FrameInCol
        if NumFlag >= SampleFrameNum and NumFlag % 30 == 0:
            flag = True
            u, s, v = lg.svd(FrameSamples, full_matrices=False)
            s = np.where(s < pca, 0, s)
            rebuild = np.dot(u * s, v)
            MeanRe = np.mean(rebuild, 1)
            rePic = np.uint8(np.reshape(MeanRe, (frame.shape[0], frame.shape[1])))
            rebuild = np.uint8(rebuild)
            cv2.imshow("rePic", rePic)
            cv2.imshow("Frames Sample with PCA", rebuild)
            cv2.imwrite(r"result\rePic" + str(NumFlag) + ".jpg", rePic)
        if NumFlag >= SampleFrameNum and flag == 1:
            Sub1 = cv2.absdiff(rePic, frame)
            ret, Sub = cv2.threshold(Sub1, 30, 255, type=cv2.THRESH_BINARY)
            cv2.imshow("Sub", Sub)
        cv2.imshow("Current frame", frame)
        cv2.imshow("Frames Sample", FrameSamples)
        cv2.waitKey(1)

funcPCA()

path1 = r'D:\Programming\MatrixTheory\IMG_0107.JPG'
path2 = r"D:\Programming\MATLAB\video_prog\org.png"
a = cv2.imread(path2)
a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
[r, c] = a.shape
r = int(r / 2)
c = int(c / 2)
I = cv2.resize(a, (c, r))
a = I
cv2.namedWindow("org", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("rebuild", cv2.WINDOW_KEEPRATIO)
for i in range(20):
    u, s, v = lg.svd(a, full_matrices=False)
    s = np.where(s < 50 + i * 400, 0, s)
    rebuild = np.uint8(np.dot(u * s, v))
    pca = 50 + i * 400
    print(50 + i * 400)
    cv2.imshow("org", a)
    cv2.imshow("rebuild", rebuild)
    cv2.imwrite("org.jpg", a)
    cv2.imwrite(r"result\rebuild" + str(pca) + ".jpg", rebuild, (r, c))
    cv2.waitKey(10)
