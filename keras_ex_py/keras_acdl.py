# -*- coding:utf-8 -*-

# 同样是imdb 数据集，但是这个是从原始数据集进行数据预处理，然后跑模型，最后预测
import keras
import os
import numpy as np
import re
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.utils import plot_model
import matplotlib.pyplot as plt

Reg = re.compile(r'[A-Za-z]*')
stop_words = ['is', 'the', 'a']

max_features = 5000
word_embedding_size = 50
maxlen = 400
filters = 250
kernel_size = 3
hidden_dims = 250


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def prepross(file):
    with open(file, encoding='utf-8') as f:
        data = f.readlines()
        data = Reg.findall(data[0])
        # 将句子中的每个单词转化为小写
        data = [x.lower() for x in data]
        # 将句子中的部分词从停用词表中剔除
        data = [x for x in data if x != '' and x not in stop_words]
        # 返回值必须是个句子，不能是单词列表
        return ' '.join(data)


def imdb_load(type):
    root_path = "E:/nlp_data/aclImdb_v1/aclImdb/"
    # 遍历所有文件
    file_lists = []
    pos_path = root_path + type + "/pos/"
    for f in os.listdir(pos_path):
        file_lists.append(pos_path + f)
    neg_path = root_path + type + "/neg/"
    for f in os.listdir(neg_path):
        file_lists.append(neg_path + f)
    # file_lists中前12500个为pos，后面为neg，labels与其保持一致
    labels = [1 for i in range(12500)]
    labels.extend([0 for i in range(12500)])
    # 将文件随机打乱，注意file与label打乱后依旧要通过下标一一对应。
    # 否则会导致 file与label不一致
    index = np.arange(len(labels))
    np.random.shuffle(index)
    # 转化为numpy格式
    labels = np.array(labels)
    file_lists = np.array(file_lists)
    labels[index]
    file_lists[index]
    # 逐个处理文件
    sentenses = []
    for file in file_lists:
        # print(file)
        sentenses.append(prepross(file))
    return sentenses, labels


def imdb_load_data():
    x_train, y_train = imdb_load("train")
    x_test, y_test = imdb_load("test")
    # 建立单词和数字映射的词典
    token = text.Tokenizer(num_words=max_features)
    token.fit_on_texts(x_train)
    # 将影评映射到数字
    x_train = token.texts_to_sequences(x_train)
    x_test = token.texts_to_sequences(x_test)
    # 让所有影评保持固定长度的词数目
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return (x_train, y_train), (x_test, y_test)


def train():
    (x_train, y_train), (x_test, y_test) = imdb_load_data()
    model = keras.Sequential()
    # 构造词嵌入层
    model.add(keras.layers.Embedding(input_dim=max_features, output_dim=word_embedding_size, name="embedding"))
    # 通过layer名字获取layer的信息
    print(model.get_layer(name="embedding").input_shape)
    # 基于词向量的堆叠方式做卷积
    model.add(keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1
                                  , activation=keras.activations.relu, name="conv1d"))
    # 对每一个卷积出的特征向量做最大池化
    model.add(keras.layers.GlobalAvgPool1D(name="maxpool1d"))
    # fc,输入是250维，输出是hidden_dims
    model.add(keras.layers.Dense(units=hidden_dims, name="dense1"))
    # 添加激活层
    model.add(keras.layers.Activation(activation=keras.activations.relu, name="relu1"))
    # fc，二分类问题，输出维度为1
    model.add(keras.layers.Dense(units=1, name="dense2"))
    # 二分类问题，使用sigmod函数做分类器
    model.add(keras.layers.Activation(activation=keras.activations.sigmoid, name="sigmoe"))
    # 打印模型各层layer信息
    model.summary()
    # 模型编译，配置loss，optimization
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    # 模型训练
    '''
    # 如果想保存每一个batch的loss等数据，需要传递一个callback
    history = LossHistory()
    train_history = model.fit(x=x_train,
                              y=y_train,
                              batch_size=128,
                              epochs=1,
                              validation_data=(x_test,y_test),
                              callbacks=[history])
    show_train_history2(history)
    # 结果可视化

    '''
    # fit 返回的log中，有 epochs 组数据，即只保存每个epoch的最后一次的loss等值
    train_history = model.fit(x=x_train,
                              y=y_train,
                              batch_size=128,
                              epochs=10,
                              validation_data=(x_test, y_test))
    show_train_history(train_history)

    # 模型保存
    model.save(filepath="./models/demo_imdb_rnn.h5")
    # 模型保存一份图片
    plot_model(model=model, to_file="./models/demo_imdb_rnn.png",
               show_layer_names=True, show_shapes=True)


def show_train_history2(history):
    plt.plot(history.losses)
    plt.title("model losses")
    plt.xlabel('batch')
    plt.ylabel('losses')
    plt.legend()
    # 先保存图片，后显示，不然保存的图片是空白
    plt.savefig("./models/demo_imdb_rnn_train.png")
    plt.show()


def show_train_history(train_history):
    print(train_history.history.keys())
    print(train_history.epoch)
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    plt.plot(train_history.history['loss'])
    plt.plot(train_history.history['val_loss'])
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


def gen_predict_data(path):
    sent = prepross(path)
    x_train, t_train = imdb_load("train")
    token = text.Tokenizer(num_words=max_features)
    token.fit_on_texts(x_train)
    x = token.texts_to_sequences([sent])
    x = sequence.pad_sequences(x, maxlen=maxlen)
    return x


RESULT = {1: 'pos', 0: 'neg'}


def predict(path):
    x = gen_predict_data(path)
    model = keras.models.load_model("./models/demo_imdb_rnn.h5")
    y = model.predict(x)
    print(y)
    y = model.predict_classes(x)
    print(y)
    print(RESULT[y[0][0]])


# train()
predict(r"E:\nlp_data\aclImdb_v1\aclImdb\test\neg\0_2.txt")
predict(r"E:\nlp_data\aclImdb_v1\aclImdb\test\pos\0_10.txt")