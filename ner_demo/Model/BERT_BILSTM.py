"""
采用 BERT + BILSTM + CRF 网络进行处理
"""


import keras_bert
# from Public.path import path_bert_dir
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Activation, Input, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
import os, tensorflow as tf
from matplotlib import pyplot
from Public.utils import *
import keras_metrics as km

from tensorflow.python.keras.utils.vis_utils import plot_model

path_bert_dir = r"/Users/zhuyingjun/Desktop/fengkong/ner_demo/data/data/uncased_L-12_H-768_A-12"


class BERTBILSTM(object):
    def __init__(self,
                 vocab_size: int,
                 n_class: int,
                 max_len: int = 128,
                 embedding_dim: int = 128,
                 rnn_units: int = 128,
                 drop_rate: float = 0.5,
                 ):

        self.n_class = n_class
        self.max_len = max_len
        self.config_path = os.path.join(path_bert_dir, 'bert_config.json')
        self.check_point_path = os.path.join(path_bert_dir, 'bert_model.ckpt')
        self.dict_path = os.path.join(path_bert_dir, 'vocab.txt')

    def creat_model(self):
        print('load bert Model start!')
        model = keras_bert.load_trained_model_from_checkpoint(self.config_path,
                                                              checkpoint_file=self.check_point_path,
                                                              seq_len=self.max_len,
                                                              trainable=False)
        print('load bert Model end!')
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = model([x1_in, x2_in])
        cls_layer = Lambda(lambda x: x[:, 0])(x)
        x = Dense(self.n_class, activation="softmax")(cls_layer)
        self.model = Model(inputs=[x1_in, x2_in], outputs=x)
        self.model.summary()
        self.compile()

        return self.model

    def compile(self):
        self.model.compile(optimizer=Adam(1e-5),
                           loss="categorical_crossentropy",
                           metrics=['acc', tf.keras.metrics.AUC(name="auc"),
                                    tf.keras.metrics.Recall(name='recall'),
                                    ])


if __name__ == '__main__':
    from DataProcess.process_data import DataProcess

    max_len = 200
    dp = DataProcess(data_type='data', model='bert', max_len=max_len)
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    md = BERTBILSTM(vocab_size=dp.vocab_size, n_class=dp.tag_size, max_len=max_len)
    md.creat_model()
    model = md.model

    plot_model(model, to_file='picture/BERT.png', show_shapes=True)

    callback = TrainHistory(log=None, model_name='BERT', model=model)  # 自定义回调 记录训练数据
    history = model.fit(train_data, train_label, batch_size=64, epochs=2,
                        validation_data=(test_data, test_label),
                        callbacks=[callback])

    pyplot.plot(history.history['auc'], label='auc')
    pyplot.plot(history.history['acc'], label='acc')
    pyplot.plot(history.history['recall'], label='recall')
    pyplot.plot(history.history['loss'], label='loss')
    pyplot.plot(history.history['val_loss'], label='val_loss')
    pyplot.plot(history.history['val_auc'], label='val_auc')

    pyplot.legend()
    pyplot.show()
    exit()
    from keras_bert import get_custom_objects

    # custom_objects = get_custom_objects()
    # my_objects = {"auc": tf.keras.metrics.AUC(name="auc"),
    #               'recall': tf.keras.metrics.Recall(name='recall')}
    # custom_objects.update(my_objects)

    model = tf.keras.models.load_model(r"/Users/zhuyingjun/Desktop/fengkong/ner_demo/save_model/BERT/1200/2.h5", custom_objects=get_custom_objects())

    # results = model.evaluate(train_data, train_label)
    # print(results)
    # exit()
    import math
    from tqdm import tqdm
    import pandas as pd, numpy as np
    exit()
    pre = model.predict(train_data)
    print(pre)
    print(pre.shape)
    pre = np.array(pre)

    print("pre shape : ", pre.shape)
    test_label = np.array(train_label)

    f = pd.DataFrame(pre, columns=["0", "1"])
    f.to_csv("./train_data_reuslt.csv", index=False)
    exit()
    df = f.sort_values(by=["1"])
    matli(df, "BiLstm Test sample num ")

    pre = model.predict(test_data)
    print(pre)
    print(pre.shape)
    pre = np.array(pre)
    print("pre shape : ", pre.shape)
    test_label = np.array(train_label)

    f = pd.DataFrame(pre, columns=["0", "1"])
    df = f.sort_values(by=["1"])
    matli(df, "BiLstm Val sample num ")
