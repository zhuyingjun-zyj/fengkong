"""
IDCNN(空洞CNN) 当卷积Conv1D的参数dilation_rate>1的时候，便是空洞CNN的操作
"""

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, Conv1D, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot
from Public.utils import *
import keras_metrics as km
import os
from Public.path import max_len


class IDCNNCRF2():
    def __init__(self,
                 vocab_size: int,  # 词的数量(词表的大小)
                 n_class: int,  # 分类的类别(本demo中包括小类别定义了7个类别)

                 embedding_dim: int = 128,  # 词向量编码长度
                 drop_rate: float = 0.5,  # dropout比例
                 ):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.drop_rate = drop_rate

    def creat_model(self):
        """
        本网络的机构采用的是，
           Embedding
           直接进行2个常规一维卷积操作
           接上一个空洞卷积操作
           连接2个全连接层
           最后连接CRF层

        kernel_size 采用2、3、4
        cnn  特征层数: 256、256、512

        """
        # word_vectors = np.array(...)  # word2vec向量的NumPy数组
        #
        # # 转换NumPy数组为TF张量
        # word_vectors_tensor = tf.convert_to_tensor(word_vectors, dtype=tf.float32)

        # 定义输入层，使用word_index作为索引来获取word2vec向量

        inputs = Input(shape=(self.max_len,), name='input_word_index')

        encoded_input = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
        # encoded_input = Embedding(
        #     input_dim=word_vectors_tensor.shape[0],  # 词汇表大小
        #     output_dim=word_vectors_tensor.shape[1],  # word2vec向量的维度
        #     weights=[word_vectors_tensor],  # 预训练的向量
        #     trainable=False  # 不需要训练这些向量
        # )(inputs)

        x = Conv1D(filters=256,
                   kernel_size=2,
                   activation='relu',
                   padding='same')(encoded_input)
        x = Conv1D(filters=256,
                   kernel_size=3,
                   activation='relu',
                   padding='same')(x)
        x = Conv1D(filters=512,
                   kernel_size=4,
                   activation='relu',
                   padding='same')(x)

        x = GlobalMaxPool1D()(x)
        drop = Dropout(self.drop_rate)(x)

        # 输出层第一个参数2是分类类别数
        outputs = Dense(2, activation='softmax')(drop)

        self.model = Model(inputs=inputs, outputs=outputs)

        self.model.summary()
        self.compile()

        return self.model

    def compile(self):
        self.model.compile(optimizer=Adam(1e-5),
                           loss="categorical_crossentropy",
                           metrics=['accuracy', tf.keras.metrics.AUC(name="auc"),
                                    tf.keras.metrics.Recall(name='recall'),
                                    ])


if __name__ == '__main__':
    from DataProcess.process_data import DataProcess
    from sklearn.metrics import f1_score
    import numpy as np
    from keras.utils.vis_utils import plot_model

    dp = DataProcess(max_len=100, data_type='msra')
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    lstm_crf = IDCNNCRF2(vocab_size=dp.vocab_size, n_class=7, max_len=100)

    lstm_crf.creat_model()

    model = lstm_crf.model

    plot_model(model, to_file='picture/IDCNN_CRF_2.png', show_shapes=True)
    exit()
