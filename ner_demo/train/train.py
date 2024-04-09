# from Model.BERT_BILSTM import BERTBILSTM
# from Model.BILSTM_Attetion_CRF import BILSTMAttentionCRF
# from Model.BILSTM import BILSTM
# from Model.IDCNN import IDCNN
from Model.IDCNN5_CRF import IDCNNCRF2
from Public.utils import *
from Public.path import *
from tensorflow.keras.callbacks import EarlyStopping
from DataProcess.process_data import DataProcess
import tensorflow as tf
print(tf.version)

def train_sample(train_model='IDCNN2',
                 epochs=30):
    # bert需要不同的数据参数 获取训练和测试数据

    dp = DataProcess(model=train_model)
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    model_class = IDCNNCRF2(dp.vocab_size, dp.tag_size)

    model = model_class.creat_model()

    callback = TrainHistory(model_name=train_model, class_model=model)  # 自定义回调 记录训练数据
    # early_stopping = EarlyStopping(monitor='val_crf_viterbi_accuracy', patience=2, mode='max')  # 提前结束
    model.fit(train_data, train_label, batch_size=128, epochs=epochs,
              validation_data=[test_data, test_label],
              callbacks=[callback])


if __name__ == '__main__':
    train_sample(train_model="IDCNN2", epochs=3)
