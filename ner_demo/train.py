from Model.BERT_BILSTM import BERTBILSTM
from Model.BILSTM_Attetion_CRF import BILSTMAttentionCRF
from Model.BILSTM import BILSTM
from Model.IDCNN import IDCNN
from Model.IDCNN5_CRF import IDCNNCRF2
from Public.utils import *
from tensorflow.keras.callbacks import EarlyStopping
from DataProcess.process_data import DataProcess

max_len = 128


def train_sample(train_model='BERTBILSTMCRF',
                 # ['BERTBILSTMCRF', 'BILSTMAttentionCRF', 'BILSTMCRF',
                 # 'IDCNNCRF', 'IDCNNCRF2']
                 epochs=15,
                 log=None,
                 ):
    # bert需要不同的数据参数 获取训练和测试数据

    dp = DataProcess(data_type='msra', max_len=max_len, model='bert')
    train_data, train_label, test_data, test_label = dp.get_data(one_hot=True)

    log.info("----------------------------数据信息 START--------------------------")
    log.info(f"当前使用数据集 MSRA")
    log.info(f"train_label:{train_label.shape}")
    log.info(f"test_label:{test_label.shape}")
    log.info("----------------------------数据信息 END--------------------------")

    if train_model == 'BERT':
        model_class = BERTBILSTM(dp.vocab_size, dp.tag_size, max_len=max_len)
    elif train_model == 'BILSTMAttention':
        model_class = BILSTMAttentionCRF(dp.vocab_size, dp.tag_size)
    elif train_model == 'BILSTM':
        model_class = BILSTM(dp.vocab_size, dp.tag_size)
    elif train_model == 'IDCNN':
        model_class = IDCNN(dp.vocab_size, dp.tag_size, max_len=max_len)
    else:
        model_class = IDCNNCRF2(dp.vocab_size, dp.tag_size, max_len=max_len)

    model = model_class.creat_model()

    callback = TrainHistory(log=log, model_name=train_model)  # 自定义回调 记录训练数据
    # early_stopping = EarlyStopping(monitor='val_crf_viterbi_accuracy', patience=2, mode='max')  # 提前结束
    model.fit(train_data, train_label, batch_size=32, epochs=epochs,
              validation_data=[test_data, test_label],
              callbacks=[callback])


if __name__ == '__main__':
    train_sample(train_model="BERT", epochs=15)
