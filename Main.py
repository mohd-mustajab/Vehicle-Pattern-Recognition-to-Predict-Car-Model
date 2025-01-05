from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
import cv2
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

main = Tk()
main.title("Vehicle Pattern Recognition using Machine & Deep Learning to Predict Car Model")
main.geometry("1300x1200")

global filename
global X, Y
global model
global X_train, X_test, y_train, y_test
accuracy = []
global XX
global classifier 

names = ['AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012', 'Acura TL Type-S 2008', 'Acura TSX Sedan 2012']

def uploadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,'dataset loaded\n')
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    X = np.asarray(X)
    Y = np.asarray(Y)
    img = X[20].reshape(64,64,3)
    cv2.imshow('ff',cv2.resize(img,(250,250)))
    cv2.waitKey(0)


def linearKNN():
    accuracy.clear()
    text.delete('1.0', END)
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    print(X.shape)
    print(Y.shape)
    temp = X
    XX = np.reshape(temp, (temp.shape[0],(temp.shape[1]*temp.shape[2]*temp.shape[3])))
    pca = PCA(n_components = 180)
    XX = pca.fit_transform(XX)
    print(XX.shape)

    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)

    cls = LogisticRegression() 
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    #predict = np.argmax(predict, axis=1)
    #y_test1 = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_test,predict)*100
    accuracy.append(acc)
    text.insert(END,'Linear Regression Prediction Accuracy : '+str(acc)+"\n")
    
    cls = KNeighborsClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    accuracy.append(acc)
    text.insert(END,'KNN Prediction Accuracy : '+str(acc)+"\n")

    bars = ('Linear Regression', 'KNN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, [accuracy[0],accuracy[1]])
    plt.xticks(y_pos, bars)
    plt.show()
    plt.title('Linear Regression & KNN Accuracy Performance Graph')
    plt.show()

def SVMCNN():
    global classifier 
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    print(X.shape)
    print(Y.shape)
    temp = X
    XX = np.reshape(temp, (temp.shape[0],(temp.shape[1]*temp.shape[2]*temp.shape[3])))
    pca = PCA(n_components = 180)
    XX = pca.fit_transform(XX)
    print(XX.shape)

    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)

    cls = svm.SVC() 
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    accuracy.append(acc)
    text.insert(END,'SVM Prediction Accuracy : '+str(acc)+"\n")

    Y1 = to_categorical(Y)
    cnn = Sequential()
    cnn.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Convolution2D(32, 3, 3, activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(output_dim = 256, activation = 'relu'))
    cnn.add(Dense(output_dim = 5, activation = 'softmax'))
    print(cnn.summary())
    cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = cnn.fit(X, Y1, batch_size=16, epochs=10, shuffle=True, verbose=2)
    cnn_history = hist.history
    cnn_history = cnn_history['accuracy']
    acc = cnn_history[9] * 100
    accuracy.append(acc)
    text.insert(END,'CNN Prediction Accuracy : '+str(acc)+"\n\n")
    classifier = cnn

    bars = ('SVM','CNN')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, [accuracy[2],accuracy[3]])
    plt.xticks(y_pos, bars)
    plt.show()
    plt.title('SVM & CNN Accuracy Performance Graph')
    plt.show()
    
    
def KNNSVM():
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    print(X.shape)
    print(Y.shape)
    temp = X
    XX = np.reshape(temp, (temp.shape[0],(temp.shape[1]*temp.shape[2]*temp.shape[3])))
    pca = PCA(n_components = 180)
    XX = pca.fit_transform(XX)
    print(XX.shape)

    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)

    cls = KNeighborsClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    accuracy.append(acc)
    text.insert(END,'KNN Prediction Accuracy : '+str(acc)+"\n")

    cls = svm.SVC() 
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    accuracy.append(acc)
    text.insert(END,'SVM Prediction Accuracy : '+str(acc)+"\n")

    bars = ('KNN Inference','SVM Inference')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, [accuracy[4],accuracy[5]])
    plt.xticks(y_pos, bars)
    plt.show()
    plt.title('KNN & SVM Inference Accuracy Performance Graph')
    plt.show()

def KNNCNN():
    X = np.load("model/X.txt.npy")
    Y = np.load("model/Y.txt.npy")
    print(X.shape)
    print(Y.shape)
    temp = X
    XX = np.reshape(temp, (temp.shape[0],(temp.shape[1]*temp.shape[2]*temp.shape[3])))
    pca = PCA(n_components = 180)
    XX = pca.fit_transform(XX)
    print(XX.shape)

    X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2)

    cls = KNeighborsClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    acc = accuracy_score(y_test,predict)*100
    accuracy.append(acc)
    text.insert(END,'KNN Prediction Accuracy : '+str(acc)+"\n")

    Y1 = to_categorical(Y)
    cnn = Sequential() 
    cnn.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Convolution2D(32, 3, 3, activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(output_dim = 128, activation = 'relu'))
    cnn.add(Dense(output_dim = 5, activation = 'softmax'))
    cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(cnn.summary())
    cnn_history = cnn.fit(X, Y1, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
    cnn_history = cnn_history.history
    cnn_history = cnn_history['accuracy']
    acc = cnn_history[9] * 100
    accuracy.append(acc)
    text.insert(END,'CNN Prediction Accuracy : '+str(acc)+"\n\n")

    bars = ('KNN Inference','CNN Inference')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, [accuracy[6],accuracy[7]])
    plt.xticks(y_pos, bars)
    plt.show()
    plt.title('KNN & CNN Inference Accuracy Performance Graph')
    plt.show()
    
def predict():
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    cv2.putText(img, 'Car Model Predicted as : '+names[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Car Model Predicted as : '+names[predict], img)
    cv2.waitKey(0)
        
def graph():
    bars = ('Linear Regression', 'KNN','SVM','CNN','SVM Inference','CNN Inference','KNN Inference','CNN Inference')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, accuracy)
    plt.xticks(y_pos, bars)
    plt.show()
    plt.title('All Algorithms Accuracy Performance Graph')
    plt.show()
    
def close():
    main.destroy()
    
font = ('times', 15, 'bold')
title = Label(main, text='Vehicle Pattern Recognition using Machine & Deep Learning to Predict Car Model')
#title.config(bg='powder blue', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Cars Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


knnButton = Button(main, text="Run Linear Regression & KNN Algorithms", command=linearKNN)
knnButton.place(x=20,y=150)
knnButton.config(font=ff)

cnnButton = Button(main, text="Run SVM & CNN Algorithms", command=SVMCNN)
cnnButton.place(x=20,y=200)
cnnButton.config(font=ff)

svmButton = Button(main, text="Run KNN & SVM Algorithms", command=KNNSVM)
svmButton.place(x=20,y=250)
svmButton.config(font=ff)

kcnnButton = Button(main, text="Run KNN & CNN Algorithms", command=KNNCNN)
kcnnButton.place(x=20,y=300)
kcnnButton.config(font=ff)

predictButton = Button(main, text="Prediction Model", command=predict)
predictButton.place(x=20,y=350)
predictButton.config(font=ff)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=20,y=400)
graphButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config()
main.mainloop()
