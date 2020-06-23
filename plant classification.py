import tensorflow.keras as k
import matplotlib.pyplot as plt
import numpy as np
import sqlite3 as  sq
import shutil as sh
import os
def classify_img():
    path="C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\train\\"

    con=sq.connect("plant.db")
    exe=con.cursor()
    exe.execute(""" SELECT image_id,source FROM train  """)
    n=0
    for img_dis in exe.fetchall():
        n = n + 1
        print(n)
        if os.path.exists("C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\train\\{}".format(img_dis[1])):
         if os.path.exists("C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\train\\{}.jpg".format(img_dis[0])):
                sh.move("C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\train\\{}.jpg".format(img_dis[0]),"C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\train\\{}\\{}.jpg".format(img_dis[1],img_dis[0]))
        else:
            os.mkdir("C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\train\\{}".format(img_dis[1]))
            if os.path.exists(
                    "C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\train\\{}.jpg".format(
                            img_dis[0])):
                sh.move(
                    "C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\train\\{}.jpg".format(
                        img_dis[0]),
                    "C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\train\\{}\\{}.jpg".format(
                        img_dis[1], img_dis[0]))


def class_plant(path_):
    path="C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\train\\"
    input=100

    GEN_=k.preprocessing.image.ImageDataGenerator(zoom_range=.1,rotation_range=.1,horizontal_flip=True, rescale=1./255)
    train_set=GEN_.flow_from_directory(path,target_size=(input,input),batch_size=32)
    #

    model = k.Sequential()
    model.add(k.layers.Conv2D(32, (3, 3), input_shape=(input, input, 3), activation='relu'))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2), strides=2))  # if stride not given it equal to pool filter size
    model.add(k.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(k.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(k.layers.Flatten())
    model.add(k.layers.Dense(units=128, activation='relu'))
    model.add(k.layers.Dropout(.25))

    model.add(k.layers.Dense(units=8, activation='softmax'))

    #
    model.compile(loss="categorical_crossentropy",optimizer="adam")
    model.fit_generator(train_set,epochs=10,validation_steps=20,verbose=1)
    #

    #
    imga=k.preprocessing.image.load_img(path_,target_size=(input,input))
    img=k.preprocessing.image.img_to_array(imga)
    img=img/255
    img=np.expand_dims(img,axis=0)
    predict=model.predict_classes(img,batch_size=None)

    plt.text(20,60,predict,bbox=dict(facecolor='white', alpha=0.8))
    plt.imshow(imga)
    plt.show()


try__=[]
for try_ in os.listdir("C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\test\\"):
    print(try_)
    try__.append(try_)
print(try__)
for try_ in try__[0:]:
    class_plant("C:\\Users\\ahmed\\PycharmProjects\\untitled\\unfinished projects\\plants\\test\\{}".format(try_))

#classify_img()