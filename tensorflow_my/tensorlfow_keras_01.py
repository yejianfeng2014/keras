import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
print(tf.keras.__version__)

# 准备数据
import  numpy as np

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))


val_x = np.random.random((100,72))
val_y = np.random.random((100,10))




# 构建 简单模型

model = tf.keras.Sequential()

model.add(layers.Dense(32,activation ='relu'))

model.add(layers.Dense(32,activation ='relu'))


model.add(layers.Dense(10, activation='softmax'))


# 网络配置
# tf.keras.layers中网络配置：
#
# activation：设置层的激活函数。此参数由内置函数的名称指定，或指定为可调用对象。默认情况下，系统不会应用任何激活函数。
#
# kernel_initializer 和 bias_initializer：创建层权重（核和偏差）的初始化方案。此参数是一个名称或可调用对象，默认为 “Glorot uniform” 初始化器。
#
# kernel_regularizer 和 bias_regularizer：应用层权重（核和偏差）的正则化方案，例如 L1 或 L2 正则化。默认情况下，系统不会应用正则化函数。


model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])


model.fit(train_x, train_y, epochs=10, batch_size=100,
          validation_data=(val_x, val_y))



