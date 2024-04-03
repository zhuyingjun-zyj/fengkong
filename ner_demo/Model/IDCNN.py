"""
IDCNN(空洞CNN) 当卷积Conv1D的参数dilation_rate>1的时候，便是空洞CNN的操作
"""
import tensorflow as tf
from Public.utils import *
from tensorflow.keras.layers import Conv1D, Embedding, Dense, Dropout, Input, MaxPooling2D, Conv2D, MaxPooling1D, \
    Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.nn import sigmoid
from matplotlib import pyplot
import keras_metrics as km
import numpy as np


class IDCNN(object):
    def __init__(self,
                 vocab_size: int,  # 词的数量(词表的大小)
                 n_class: int,  # 分类的类别(本demo中包括小类别定义了2个类别)
                 max_len: int = 128,  # 最长的句子最长长度
                 embedding_dim: int = 128,  # 词向量编码长度
                 drop_rate: float = 0.4,  # dropout比例
                 ):
        self.vocab_size = vocab_size
        self.n_class = n_class
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.drop_rate = drop_rate
        pass

    def creat_model(self):
        main_input = Input(shape=(self.max_len,), dtype='float64')

        embed = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim)(main_input)
        # 卷积核个数为128，词窗大小分别为2,3,4,6
        cnn1 = Conv1D(128, 2, padding='same', strides=1, activation='relu')(embed)
        cnn1 = MaxPooling1D(pool_size=self.max_len - 1)(cnn1)
        cnn2 = Conv1D(128, 3, padding='same', strides=1, activation='relu')(embed)
        cnn2 = MaxPooling1D(pool_size=self.max_len - 2)(cnn2)
        cnn3 = Conv1D(128, 4, padding='same', strides=1, activation='relu')(embed)
        cnn3 = MaxPooling1D(pool_size=self.max_len - 3)(cnn3)
        cnn4 = Conv1D(128, 6, padding='same', strides=1, activation='relu')(embed)
        cnn4 = MaxPooling1D(pool_size=self.max_len - 5)(cnn4)
        # 合并三个模型的输出向量
        cnn = concatenate([cnn1, cnn2, cnn3, cnn4], axis=-1)
        flat = Flatten()(cnn)
        drop = Dropout(self.drop_rate)(flat)
        # 输出层第一个参数2是分类类别数
        output = Dense(2, activation='sigmoid')(drop)

        model = keras.Model(inputs=main_input, outputs=output)

        model.summary()
        self.model = model
        self.model.summary()
        self.compile()

        return self.model

    def compile(self):
        self.model.compile(optimizer=Adam(1e-5),
                           loss="categorical_crossentropy",
                           metrics=['accuracy', tf.keras.metrics.AUC(name="auc"),
                                    tf.keras.metrics.Recall(name='recall'),
                                    km.f1_score(),
                                    km.precision(),
                                    ])

    def predict(self, model, new_sen):
        model = self.creat_model()
        model.load_weights(model_path)
        new_sen_list = jieba.lcut(new_sen)
        sen2id = [self.vocab.get(word, 0) for word in new_sen_list]
        sen_input = pad_sequences([sen2id], maxlen=self.maxlen)
        res = model.predict(sen_input)[0]
        return np.argmax(res)


if __name__ == '__main__':
    from DataProcess.process_data import DataProcess
    from sklearn.metrics import f1_score, recall_score
    from Public.mat import matli
    from tensorflow.python.keras.utils.vis_utils import plot_model

    MAX_LEN = 128
    epochs = 60
    batch_size = 128
    #
    dp = DataProcess(max_len=MAX_LEN, data_type='data')
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)
    #
    # # 这里使用10000个常用单词，减少计算
    #
    # model_class = IDCNN(vocab_size=dp.vocab_size, n_class=dp.tag_size, max_len=MAX_LEN)
    # model_class.creat_model()
    # model = model_class.model
    # plot_model(model, to_file='./picture/IDCNN.png', show_shapes=True)
    #
    # callback = TrainHistory(log=None, model_name='IDCNN', model=model)  # 自定义回调 记录训练数据
    # # early_stopping = EarlyStopping(monitor='val_crf_viterbi_accuracy', patience=2, mode='max')  # 提前结束
    # history = model.fit(train_data, train_label, batch_size=batch_size, epochs=epochs,
    #                     validation_data=[test_data, test_label],
    #                     callbacks=[callback])
    #
    # pyplot.plot(history.history['auc'], label='auc')
    # pyplot.plot(history.history['acc'], label='acc')
    # pyplot.plot(history.history['recall'], label='recall')
    # pyplot.plot(history.history['f1_score'], label='f1_score')
    # pyplot.plot(history.history['precision'], label='precision')
    # pyplot.plot(history.history['loss'], label='loss')
    # pyplot.plot(history.history['val_loss'], label='val_loss')
    # pyplot.plot(history.history['val_auc'], label='val_auc')
    # pyplot.plot(history.history['val_recall'], label='val_recall')
    # pyplot.plot(history.history['val_f1_score'], label='val_f1_score')
    #
    # pyplot.legend()
    # pyplot.show()


    model = tf.keras.models.load_model(r"/Users/zhuyingjun/Desktop/fengkong/ner_demo/save_model/IDCNN/58.h5",custom_objects={"binary_f1_score":km.f1_score(),"binary_precision":km.precision()})
    import math
    from tqdm import tqdm
    import pandas as pd

    results = model.evaluate(test_data, test_label)
    print(results)
    exit()
    pre = model.predict(test_data)
    print(pre)
    print(pre.shape)
    pre = np.array(pre)
    print("pre shape : ", pre.shape)
    test_label = np.array(train_label)

    f = pd.DataFrame(pre, columns=["0", "1"])
    df = f.sort_values(by=["1"])
    df.to_csv('cnn_train_54_区分度.csv')

    n = 10
    y1 = []
    df_num = len(df)
    every_epoch_num = math.floor((df_num / n))
    for index in tqdm(range(n)):
        if index < n - 1:
            df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
        else:
            df_tem = df[every_epoch_num * index:]
        col_mean = df_tem.mean(axis=0)
        print(type(col_mean))
        y1.append('%.3f' % col_mean["1"])
        print(col_mean["1"])

    matli(df, "CNN Val sample num ")


    # pre = pre.reshape(pre.shape[0] * pre.shape[1], )
    # test_label = test_label.reshape(test_label.shape[0] * test_label.shape[1], )
#
#     f1score = f1_score(pre, test_label, average='macro',zero_division="warn")
#     recall = recall_score(pre, test_label, average='macro',zero_division="warn")
#
#     logging.info("================================================")
#     logging.info(f"--------------:f1: {f1score} --------------")
#     logging.info(f"--------------:recall: {recall} --------------")
#     logging.info("================================================")
# # exit()
#
# from Public.path import path_log_dir, path_model_idcnn
#
# save_path = os.path.join(path_model_idcnn, str(25) + ".h5")
# # 38 是最好的
#
# model.load_weights(save_path)
# from sklearn.metrics import f1_score, recall_score, precision_score
# import pandas as pd,numpy as np
#
# # # 对比测试数据的tag
# pridict_indexs = model.predict(test_data)
# print("pridict_indexs shape: ", pridict_indexs.shape)
#
# pridict_indexs = np.argmax(pridict_indexs,axis=1)
# print("pridict_indexs shape: ", pridict_indexs.shape)
#
# print("shape:", test_data.shape, "type:", type(test_data))
# print("test_label shape:", test_label.shape, "type:", type(test_label))
# test_label = test_label.reshape((len(test_label),))
# print("test_label shape:", test_label.shape), "type:", type(test_label)
#
# f1score = f1_score(test_label, pridict_indexs)
# re_score = recall_score(test_label, pridict_indexs)
#
# pre_score = precision_score(test_label, pridict_indexs)
# print(f"f1score:{f1score}   re_score:{re_score}   pre_score:{pre_score}")
#
# with open('./pre21.txt', 'w') as f:
#     f.write("".join(texts))

# address = "安徽省池州市石台县秋浦东路１６号中国电信石台县秋浦路营业厅"
# address2id = dp.predict_text_to_index(address)
# print("address2id: ", address2id, "shape:", address2id.shape, "type:", type(address2id))
# labels = model.predict(address2id)[0]
# print("labels: ", labels)
# labels = np.argmax(labels,1)[:len(list(address))]
# print("labels: ", labels)
# r = [dp.num2tag().get(i) for i in labels]
# print(r)
