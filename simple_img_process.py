import sys 
import cv2
import numpy as np
from source.makeLayers import makeLayers
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ノーツを加える [simple_img_process.py]
def addNote(src_img,center,radius,color=None):
    if color is None: color = (70,70,70)
    tar_img = cv2.circle(src_img,center=center,radius=radius,color=color,thickness=-1)
    return tar_img

# 疑似画面キャプチャデータの生成 [simple_img_process.py]
def generateImg(data_n=100):
    w = 80
    h = 60
    c = 3
    bg_color = (180,220,220)
    na = np.array
    x_ls = []
    y_ls = []
    for ind_note in range(5):
        for i in range(data_n):
            img = np.ones((h,w,c),dtype="uint8")
            img[:,:,0] = bg_color[0]
            img[:,:,1] = bg_color[1]
            img[:,:,2] = bg_color[2]
            for j in range(5):
                center = (10+15*j,50)
                img = addNote(img,center=center,radius=6,color=(200,200,200))
            note_y = int(0.5*i)
            img = addNote(img,center=(10+15*ind_note,note_y),radius=6,color=(230,160,160))
            if note_y > 40:
                y = [1 if ind_note==k else 0 for k in range(5)]
            else:
                y = [0,0,0,0,0]
            y_ls.append(y)
            x_ls.append(img)
    x_ls = na(x_ls)
    y_ls = na(y_ls)
    return x_ls, y_ls

# 議事画面キャプチャデータの表示 [simple_img_process.py]
def showSamples(x_ls,y_ls,show_n=None):
    import matplotlib.pyplot as plt
    if (show_n is None) or (show_n>len(x_ls)):
        show_n = len(x_ls)
    for i in range(show_n):
        plt.imshow(x_ls[i])
        plt.title("ind=%d,label=%d"%(i,y_ls[i][2]))
        plt.pause(0.05)    


if __name__ == '__main__':
    # 疑似画面キャプチャデータの生成 [simple_img_process.py]
    x_ls, y_ls = generateImg()
    data_n = len(x_ls)
    #行列のシャッフル
    order_ls = list(range(data_n))
    random.shuffle(order_ls)
    x_ls = np.array([x_ls[ind] for ind in order_ls])
    y_ls = np.array([y_ls[ind] for ind in order_ls])
    showSamples(x_ls,y_ls,100)

    #セット分割
    #x_train, x_test, y_train, y_test = splitData(x_ls,y_ls)
    x_train = x_ls[:int(data_n/2)]
    x_test  = x_ls[int(data_n/2):]
    y_train = y_ls[:int(data_n/2)]
    y_test  = y_ls[int(data_n/2):]

    #モデル定義
    model = makeLayers(x_ls.shape[1:],5)
    #train
    model.fit(x_train, y_train,batch_size=32,epochs=100,verbose=1,validation_data=(x_test,y_test))
    #evaluation
    y_pred = model.predict(x_test)
    y_pred = [np.argmax(one_y) for one_y in y_pred]
    y_test = [np.argmax(one_y) for one_y in y_test]
    confmat = confusion_matrix(y_test, y_pred, labels=list(range(5)))
    plt.imshow(confmat)
    plt.show()