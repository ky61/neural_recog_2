import sys 
import cv2
import numpy as np

# ノーツを加える
def addNote(src_img,center,radius,color=None):
    if color is None: color = (70,70,70)
    tar_img = cv2.circle(src_img,center=center,radius=radius,color=color,thickness=-1)
    return tar_img

# 疑似画面キャプチャデータの生成 [simple_img_process.py]
def generateImg(data_n=100):
    w = 80
    h = 60
    c = 3
    na = np.array
    x_ls = []
    y_ls = []
    for i in range(data_n):
        if i%2 == 0:
            img = np.zeros((h,w,c),dtype="uint8")
            y_ls.append(0)
        else:
            img = np.ones((h,w,c),dtype="uint8")*255
            center = (40,30)
            color = (230,160,160)
            img = addNote(img,center=center,radius=7,color=color)
            y_ls.append(1)
        x_ls.append(img)
    x_ls = na(x_ls)
    y_ls = na(y_ls)
    return x_ls, y_ls

# 議事画面キャプチャデータの表示 [simple_img_process.py]
def showSamples(x_ls,y_ls):
    import matplotlib.pyplot as plt
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(x_ls[i])
        plt.title(y_ls[i])
    plt.show()    


if __name__ == '__main__':
    # 疑似画面キャプチャデータの生成 [simple_img_process.py]
    x_ls, y_ls = generateImg()
    showSamples(x_ls,y_ls)
    sys.exit()
    #セット分割
    x_train, x_test, y_train, y_test = splitData(x_ls,y_ls)
    #モデル定義
    model = makeLayer()
    #train
    model.fit(x_train, y_train)
    #evaluation
    score = model.evaluate(x_test,y_test)
    print(score)