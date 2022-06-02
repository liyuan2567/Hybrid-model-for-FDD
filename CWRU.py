import numpy as np
import pandas as pd
import os
from skimage import img_as_ubyte
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.applications import resnet
from keras.utils.np_utils import to_categorical
from keras import optimizers
import cv2 as cv
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import pickle
from sklearn.metrics import classification_report,precision_score,recall_score,accuracy_score,f1_score
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization,Dropout,Activation
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import math
import warnings
warnings.filterwarnings("ignore")

class HybridModel():
    def __init__(self, model_2):
        self.N = 32
        self.num_cla = 16
        self.prec_threshold = 0.9
        self.threshold_proba_everyclass = 0.6
        self.f1_threshold = None
        self.model_1 = ['rf','svm','knn']
        self.model_2 = model_2
        self.Load = ['Load_0','Load_1','Load_2','Load_3']
        self.window_step = 700
        self.path_model = './Model'
        self.path_data = './thesis_data'
        self.path_tra_His = './History'
        self.probability_threshold = [0.2,0.4,0.25,0.3]

    def sliding_window(self, num, step):
        rs = []
        for i in range(len(num)-(step-1)):
          rs.append(sum(num[i:i+step])/step)
        return rs

    def data_processing(self,path_data,Load,window_step):
        # 文件夹目录
        path = f"{self.path_data}/{Load}"
        # 得到文件夹下的所有文件名称
        files = os.listdir(path)
        # 初始化类别的个数
        # num_class = 0
        #初始化label的list
        y_data = []
        target_names = []
        x_data = []
        # 遍历文件夹
        for file in files:
            # There are hidden .DS_Store files in each folder of MAC, which can be removed manually
            if file !=".DS_Store":

                pd_reader = pd.read_csv(path + "/" + file)
                pd_de = np.array(pd_reader["DE"])
                pd_de = self.sliding_window(pd_de,3)
                #标准化        
                pd_de_norm = img_as_ubyte((pd_de - min(pd_de)) / (max(pd_de) - min(pd_de)))  
                x_data_every_class = []
                i = 0
                window_step = window_step
                while window_step*i + (self.N*self.N) < len(pd_de_norm): 
                    x_data_every_class.append(pd_de_norm[window_step*i:window_step*i+self.N*self.N])
                    i+=1
                # 每个类别可以生成多少张图片，在Load0,normal:59,其他都是29；load1-3，normal：118，其他都是29
                num_label_train = len(x_data_every_class)
                x_data = x_data + x_data_every_class
                #每个类别可以生成
                # num_class += 1

                if Load == 'Load_all':
                    target_names.append(os.path.basename(path + "/" + file)[:-6])
                    label_everycsv = [os.path.basename(path + "/" + file)[:-6]] * num_label_train
                else:
                    target_names.append(os.path.basename(path + "/" + file)[:-4])
                    label_everycsv = [os.path.basename(path + "/" + file)[:-4]] * num_label_train
                #做label
                y_data = y_data + label_everycsv
        #label string--int
        label = LabelEncoder()
        y_data_num = label.fit_transform(y_data)
        #找int和string的对应关系,即l1和l3的对应关系
        l1 = sorted(set(y_data),key=y_data.index)
        l2 = list(y_data_num)
        l3 = sorted(set(l2),key=l2.index)
        class_dic = dict(zip(l3, l1))
        x_data = (np.array(x_data)).reshape(len(x_data),self.N*self.N)
        print(f'Load {Load} totally has {x_data.shape[0]} {x_data.shape[1] ** 0.5}*{x_data.shape[1] ** 0.5} pictures.')
        print(f'Load {Load} has {self.num_cla} classes.') 
        return x_data, l2, class_dic

    #ml算法数据格式
    def train_test_data(self,path_data,Load,window_step):
        x_data, y_data, class_dic = self.data_processing(path_data,Load,window_step)
        x_train_val, x_test, y_train_val, y_test = train_test_split(x_data, y_data, test_size=.17, stratify = y_data, random_state=0)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=.2, stratify = y_train_val, random_state=0)
        print(f'Load {Load} has {x_train.shape[0]} {int((x_train.shape[1]) ** 0.5)}*{int((x_train.shape[1]) ** 0.5)} training pictures.')
        print(f'Load {Load} has {x_val.shape[0]} {int((x_val.shape[1]) ** 0.5)}*{int((x_val.shape[1]) ** 0.5)} validation pictures.')
        print(f'Load {Load} has {x_test.shape[0]} {int((x_test.shape[1]) ** 0.5)}*{int((x_test.shape[1]) ** 0.5)} testing pictures.')
        return x_train_val, x_train, x_val, x_test, y_train_val, y_train, y_val, y_test

    # 训练模型
    #ml-model 训练
    def Training_model_1(self,model_1, Load, x_train, y_train, x_val, y_val):
        #SVM
        start_training_model_1 = time.time()
        if model_1 == 'svm':
            model = OneVsRestClassifier(SVC(C=3.89, kernel='rbf', probability=True, random_state=None))
        elif model_1 =='knn':
            model = KNeighborsClassifier(n_neighbors=10)
        elif model_1 =='rf':
            model = RandomForestClassifier(n_estimators=250)
        
        print(f"[INFO] Successfully initialize {model_1} model !")
        print("[INFO] Training the model…… ")
        clt = model.fit(x_train, y_train)
        print("[INFO] Model training completed !")
        end_training_model_1 = time.time()
        duration_training_model_1 = end_training_model_1-start_training_model_1
        print(f'Training time of {Load} {model_1}:%s Seconds'%(duration_training_model_1))
        # save the model to disk
        filename = f'{self.path_model}/{model_1}_{Load}.sav'
        pickle.dump(model, open(filename, 'wb'))
        y_val_pred = model.predict(x_val)
        t1 = classification_report(y_val, y_val_pred, output_dict=True)
        t2 = classification_report(y_val, y_val_pred, output_dict=False)
        # print(t2)
        return t1, duration_training_model_1


    #训练过程中用val_data筛选哪些数据要去model_2
    def Data_tomodel2(self,model_2,t1,prec_threshold,recall_threshold,x_train_val,y_train_val):
        class_notto_model2 = []
        class_list = list(t1.keys())[:self.num_cla]
        prfs_list = list(t1.values())[:self.num_cla]
        for i in range(len(class_list)):
            prfs_everyclass_dict = prfs_list[i]
            prec = prfs_everyclass_dict['precision']
            recall = prfs_everyclass_dict['recall']
            # f1 = prfs_everyclass_dict['f1-score']
            # if prec >= prec_threshold and recall >= recall_threshold and f1 >= f1_threshold:
            if prec >= prec_threshold and recall >= recall_threshold:
                class_notto_model2.append(int(class_list[i]))  
      
        #训练阶段_选出送进model2的数据
        pd_x_train = pd.DataFrame(x_train_val,index=y_train_val)
        pd_xy_train = pd_x_train.drop(class_notto_model2)
        x_model_2 = pd_xy_train.values
        if model_2 == 'NN':
            x_model_2 = x_model_2.reshape(x_model_2.shape[0],self.N,self.N)
        y_model_2 = list(pd_xy_train.index)
        return x_model_2,y_model_2,class_notto_model2

    def adaptive_threshold(self,y_train):
        most = dict(Counter(y_train))
        distibution = max(most.values())/len(y_train)*100
        threshold = math.log(129*distibution, 14195)
        return threshold

    def NN_selfdef(self):
        model = keras.Sequential()
        model.add(Conv2D(64, (3,3), strides=(1,1), input_shape=(self.N,self.N,1), padding='valid', kernel_initializer= tf.keras.initializers.HeNormal()))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))
        model.add(Conv2D(128, (3,3), strides=(1,1), padding='valid', kernel_initializer=tf.keras.initializers.HeNormal()))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='softmax'))
        initial_learning_rate = 0.0015
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                      initial_learning_rate,
                      decay_steps=200,
                      decay_rate=0.96,
                      staircase=True)
        sgd = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
        return model


    def Training_model_2(self,model_1, model_2,Load, x_train, y_train, num_cla, path_tra_His):   
        if model_2 == 'NN':
            x_train = x_train.reshape(x_train.shape[0],self.N,self.N)
            y_train = to_categorical(y_train, num_classes = num_cla)
            x_train_model_2, x_test_model_2, y_train_model_2, y_test_model_2 = train_test_split(x_train, y_train, test_size=0.2,stratify = y_train, random_state=0)
            start_training_model_2 = time.time()
            model = self.NN_selfdef()
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',patience=18,mode='max',min_delta = 0.003)
            print("[INFO] Training the model…… ")
            history = model.fit(x_train_model_2,y_train_model_2, epochs=100,batch_size=64,validation_data=(x_test_model_2, y_test_model_2), callbacks=[callback])
            print("[INFO] Model training completed !")
            end_training_model_2 = time.time()
            duration_training_model_2 = end_training_model_2-start_training_model_2
            model.save(f'{self.path_model}/{model_1}_{model_2}_{Load}.h5')
            print(f"[INFO] Model has saved in '{self.path_model}/{model_1}_{model_2}_{Load}.h5'.")
            duration_model_2 = end_training_model_2-start_training_model_2
            print(f'Training time of {Load} model_2:%s Seconds'%(duration_model_2))
            with open(f'{path_tra_His}/{model_1}_{model_2}_{Load}.txt', 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
        else:
            start_training_model_2 = time.time()
            if model_2 == 'svm':
                model = OneVsRestClassifier(SVC(C=3.89, kernel='rbf', probability=True, random_state=None))
            elif model_2 =='knn':
                model = KNeighborsClassifier(n_neighbors=10)
            elif model_2 =='rf':
                model = RandomForestClassifier(n_estimators=250)
            print(f"[INFO] Successfully initialize {model_2} model !")
            print("[INFO] Training the model…… ")
            clt = model.fit(x_train, y_train)
            print("[INFO] Model training completed !")
            end_training_model_2 = time.time()
            duration_training_model_2 = end_training_model_2-start_training_model_2
            print(f'Training time of {Load} model_2: {model_2}:%s Seconds'%(duration_training_model_2))
            # save the model to disk
            filename = f'{self.path_model}/{model_1}_{model_2}_{Load}.sav'
            pickle.dump(model, open(filename, 'wb'))
        return duration_training_model_2


    def test_hybrid_model_time(self,path_model, model_1, model_2, Load, x_test, y_test, threshold_proba_everyclass):
        filename = f'{path_model}/{model_1}_{Load}.sav'
        model1 = pickle.load(open(filename, 'rb'))
        if model_2 =='NN':
            model2 = keras.models.load_model(f'{path_model}/{model_1}_{model_2}_{Load}.h5')
        else:
            filename = f'{path_model}/{model_1}_{model_2}_{Load}.sav'
            model2 = pickle.load(open(filename, 'rb'))

        start_test_model_1 = time.time()
        proba_everyclass = model1.predict_proba(x_test)
        end_test_model_1 = time.time()
        dura_test_model_1 = end_test_model_1 - start_test_model_1
        print('Inference time of machine learning model :%s Seconds'%(dura_test_model_1))
        
        start_filter = time.time()
        x_test_pd = pd.DataFrame(x_test)
        x_test_model_2 = pd.DataFrame()
        for i in range(proba_everyclass.shape[0]):
            if max(proba_everyclass[i]) <= threshold_proba_everyclass:
                #取出不符合条件的行
                x_test_model_2 = pd.concat((x_test_model_2, x_test_pd[i:i+1]), axis=0)
        x_test_model_2 = x_test_model_2.values
        print(f'There will be {x_test_model_2.shape[0]} pictures that will be fed into the deep learning model.' )
        end_filter = time.time()
        filter_time = end_filter - start_filter
        print('Filter time :%s Seconds'%(filter_time))

        start_test_model_2 = time.time()
        test_data = []
        if model_2 == 'NN':
            ##黑白
            x_test_model_2 = x_test_model_2.reshape(x_test_model_2.shape[0],self.N,self.N)
            test_out = model2.predict(x_test_model_2)
            test_out = np.argmax(test_out, axis=1)
        else:
            test_out = model2.predict(x_test_model_2)

        end_test_model_2 = time.time()
        dura_test_model_2 = end_test_model_2 - start_test_model_2
        dura_test_model_all = end_test_model_2 - start_test_model_1
        print(f'Inference time of {model_2} model :%s Seconds'%(dura_test_model_2))
        print('Inference time of hybrid-model :%s Seconds'%(dura_test_model_all))
        return dura_test_model_1, filter_time, dura_test_model_2, dura_test_model_all

    #验证阶段：重点关注！！！精确度！！！: 打印测试集每一张图片属于每一类的概率
    def test_hybrid_model_accuracy(self,path_model, model_1, model_2, Load, x_test, y_test, threshold_proba_everyclass):
        filename = f'{path_model}/{model_1}_{Load}.sav'
        model1 = pickle.load(open(filename, 'rb'))
        if model_2 =='NN':
            model2 = keras.models.load_model(f'{path_model}/{model_1}_{model_2}_{Load}.h5')
        else:
            filename = f'{path_model}/{model_1}_{model_2}_{Load}.sav'
            model2 = pickle.load(open(filename, 'rb'))
        pre_test = model1.predict(x_test)
        proba_everyclass = model1.predict_proba(x_test)
        max_proba_everyclass = np.max(proba_everyclass,axis=1)
        test_res = np.argmax(proba_everyclass, axis=1)
        x_test_pd = pd.DataFrame(x_test)
        x_test_model_1 = pd.DataFrame()
        x_test_model_2 = pd.DataFrame()
        y_test_model_1 = []
        y_test_model_2 = []
        pred_test_model_1 = []
        for i in range(proba_everyclass.shape[0]):
            if max_proba_everyclass[i] <= threshold_proba_everyclass:
                #取出不符合条件的行
                #转去dl的数据
                x_test_model_2 = pd.concat((x_test_model_2, x_test_pd[i:i+1]), axis=0)
                #转去dl的label
                y_test_model_2.append(y_test[i])
            else:
                #留在ml的数据
                # x_test_ml = pd.concat((x_test_ml, x_test_pd[i:i+1]), axis=0)
                #留在ml的label
                y_test_model_1.append(y_test[i])
                #留在ml的预测结果
                pred_test_model_1.append(pre_test[i])
        x_test_model_2 = x_test_model_2.values
        y_test_model_2_counter = Counter(y_test_model_2)
        y_test_model_1_counter = Counter(y_test_model_1)
        print(f'y_test_model_2_counter:{y_test_model_2_counter}')
        print(f'y_test_model_1_counter:{y_test_model_1_counter}')
        print(f'{x_test.shape[0]} pictures will be fed into the hybrid-model.')
        print(f'{x_test_model_2.shape[0]} pictures will be fed into the deep learning model.' )

        test_data = []
        if model_2 == 'NN':
            x_test_model_2 = x_test_model_2.reshape(x_test_model_2.shape[0],self.N,self.N)
            pred_test_model_2 = model2.predict(x_test_model_2)
            pred_test_model_2 = (np.argmax(pred_test_model_2, axis=1)).tolist()
        else:
            pred_test_model_2 = (model2.predict(x_test_model_2)).tolist()

        #整个hybrid-model approach accurancy计算: y_test_model_2 -- pre_test_model_2 ; y_test_model_1 -- pred_test_model_1
        y_test_all = y_test_model_1 + y_test_model_2
        pred_test_all = pred_test_model_1 + pred_test_model_2
        acc_hybrid_model = accuracy_score(y_test_all, pred_test_all)
        acc_model_1 = accuracy_score(y_test_model_1, pred_test_model_1)
        acc_model_2 = accuracy_score(y_test_model_2, pred_test_model_2)
        print(f'The accurancy of first model: {model_1} is {acc_model_1}')
        print(f'The accurancy of second model: {model_2} is {acc_model_2}')
        print(f'The accurancy of the hybrid-model approach is {acc_hybrid_model}')
        return acc_hybrid_model, acc_model_1, acc_model_2

    #对比实验：全量数据进入cnn
    def NN_train(self,x_train_val, y_train_val, Load, num_cla, path_model, path_tra_His):
        x_train_val = x_train_val.reshape(x_train_val.shape[0],self.N,self.N)
        y_train_val = to_categorical(y_train_val, num_classes = num_cla)
        x_train_model_2, x_test_model_2, y_train_model_2, y_test_model_2 = train_test_split(x_train_val, y_train_val, test_size=0.2,stratify = y_train_val, random_state=0)
        start_nn = time.time()
        model = self.NN_selfdef()
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',patience=18,mode='max',min_delta = 0.003)
        print("[INFO] Training the model…… ")
        history = model.fit(x_train_model_2,y_train_model_2, epochs=100,batch_size=64,validation_data=(x_test_model_2, y_test_model_2),callbacks=[callback])
        print("[INFO] Model training completed !")
        model.save(f'{path_model}/NN_{Load}.h5')
        print(f"[INFO] Model has saved in '{path_model}/NN_{Load}.h5'.")
        end_nn = time.time()
        dura_nn = end_nn - start_nn
        print(f'Training time of only NN model:%s Seconds'%(dura_nn))
        with open(f'{path_tra_His}/NN_{Load}.txt', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        return dura_nn

    # #全量数据进cnn验证--重点关注时间和精确度
    def NN_test(self,path_model, Load, x_test, y_test):
        NN_load = keras.models.load_model(f'{path_model}/NN_{Load}.h5')
        start_test = time.time()
        x_test = x_test.reshape(x_test.shape[0],self.N,self.N)
        pre_test = NN_load.predict(x_test)
        pre_test = np.argmax(pre_test, axis=1)
        end_test = time.time()
        dura_NN_test = end_test - start_test
        print(f'Inference time of only NN model :%s Seconds'%(dura_NN_test))
        acc_NN = precision_score(y_test, pre_test,average='micro')
        print(f'The accurancy of the only NN model is {acc_NN}')
        return dura_NN_test, acc_NN

    def run(self):
        duration_all = []
        acc_all = []
        c = 0
        for load in self.Load:
            # 对于每一个load，分割数据集
            x_train_val, x_train, x_val, x_test, y_train_val, y_train, y_val, y_test = self.train_test_data(self.path_data, load, 700)
            # 对于每一个load，获取数据分布
            recall_threshold = self.adaptive_threshold(y_train)
            for model1 in self.model_1:

              # 训练机器学习模型1
              t1, duration_training_model_1 = self.Training_model_1(model1, load, x_train, y_train, x_val, y_val)

              # 筛选出需要进下一模型的数据
              x_model_2,y_model_2,class_notto_model2 = self.Data_tomodel2(self.model_2,t1,self.prec_threshold,recall_threshold,x_train_val,y_train_val)

              # 开始训练模型2
              duration_training_model_2 = self.Training_model_2(model1, self.model_2,load, x_train, np.array(y_train), self.num_cla, self.path_tra_His)

              # 评估模型2时间
              if model1 == 'rf':
                threshold_proba_everyclass=self.probability_threshold[c]
              else:
                threshold_proba_everyclass=self.threshold_proba_everyclass
              dura_test_model_1, filter_time, dura_test_model_2, dura_test_model_all = self.test_hybrid_model_time(self.path_model, model1, self.model_2, load, x_test, y_test, threshold_proba_everyclass)
          
              # 评估混合模型精度
              acc_hybrid_model, acc_model_1, acc_model_2 = self.test_hybrid_model_accuracy(self.path_model, model1, self.model_2, load, x_test, y_test, threshold_proba_everyclass)
              duration_all.append([f'{load}+{model1}+{self.model_2}:',duration_training_model_1,duration_training_model_2,dura_test_model_1, filter_time, dura_test_model_2, dura_test_model_all])
              acc_all.append([f'{load}+{model1}+{self.model_2}:',acc_hybrid_model, acc_model_1, acc_model_2])
            # 对于每个load训练全量CNN+评估
            dura_nn = self.NN_train(x_train_val, np.array(y_train_val), load, self.num_cla, self.path_model, self.path_tra_His)
            dura_NN_test, acc_NN = self.NN_test(self.path_model, load, x_test, y_test)
            duration_all.append((f'{load}: ', [dura_nn, dura_NN_test]))
            acc_all.append((f'{load}: ', acc_NN)) 
            pd.DataFrame(duration_all).to_csv('./duration_banlance_mlnn.csv',index=None)
            pd.DataFrame(acc_all).to_csv('./acc_banlance_mlnn.csv',index=None)
        c+=1
        return duration_all, acc_all
     
if __name__ == '__main__':
    duration_all, acc_all = HybridModel(model_2='NN').run()
    print(duration_all)
    print(acc_all)