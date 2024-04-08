import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Bidirectional, LSTM, Embedding, Dense, Dropout, Input, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot
from Public.utils import *
import keras_metrics as km
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BILSTM():
    def __init__(self,
                 vocab_size: int,
                 n_class: int = 2,
                 embedding_dim: int = 128,
                 max_len: int = 128,
                 rnn_units: int = 128,
                 drop_rate: float = 0.5,
                 ):
        self.max_len = max_len
        self.model = None
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.drop_rate = drop_rate

    def creat_model(self):
        # model = Sequential()
        # model.add(Embedding(output_dim=self.embedding_dim,
        #                     input_dim=self.vocab_size + 1,
        #                     input_length=self.max_len))
        #
        # model.add(Bidirectional(LSTM(units=self.rnn_units), merge_mode='concat'))
        # model.add(Dropout(self.drop_rate))
        # model.add(Dense(self.n_class, activation='softmax'))
        # # model.add(Activation('softmax'))
        # # self.model = Model(inputs=inputs, outputs=x)

        inputs = Input(shape=(None,))
        x = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(inputs)
        x = Bidirectional(LSTM(units=self.rnn_units, return_sequences=True))(x)
        # x = AttentionSelf(300)(x)
        x = Dropout(self.drop_rate)(x)
        x = Dense(self.n_class, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=x)
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
    from tensorflow.python.keras.utils.vis_utils import plot_model
    from Public.mat import matli

    max_len = 200
    dp = DataProcess(data_type='data', max_len=max_len)
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    lstm_crf = BILSTM(vocab_size=dp.vocab_size, max_len=max_len)
    # lstm_crf.creat_model()
    # model = lstm_crf.model
    # #
    # plot_model(model, to_file='picture/BILSTM.png', show_shapes=True)
    # callback = TrainHistory(log=None, model_name='BILSTM', model=model)  # 自定义回调 记录训练数据
    # history = model.fit(train_data, train_label, batch_size=64, epochs=120,
    #                     validation_data=[test_data, test_label],
    #                     callbacks=[callback])
    # pyplot.plot(history.history['auc'], label='auc')
    # pyplot.plot(history.history['acc'], label='acc')
    # pyplot.plot(history.history['recall'], label='recall')
    # pyplot.plot(history.history['loss'], label='loss')
    # pyplot.plot(history.history['val_loss'], label='val_loss')
    # pyplot.plot(history.history['val_auc'], label='val_auc')
    # pyplot.savefig('save.jpg')
    # pyplot.show()
    # pyplot.legend()

    import pandas as pd, numpy as np
    import math
    from tqdm import tqdm  # 0-68,

    model = tf.keras.models.load_model(r"/Users/zhuyingjun/Desktop/fengkong/ner_demo/save_model/BILSTM/1.6w_200/50.h5")
    # results = model.evaluate(train_data, train_label)
    # print(results)
    # exit()
    import math
    from tqdm import tqdm
    import pandas as pd

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
