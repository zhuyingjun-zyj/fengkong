import logging
from tensorflow import keras
import os
from Public.path import path_log_dir
path_model = r'/Users/zhuyingjun/Desktop/fengkong/ner_demo/save_model/'

def create_log(path, stream=False):
    """
    获取日志对象
    :param path: 日志文件路径
    :param stream: 是否输出控制台
                False: 不输出到控制台
                True: 输出控制台，默认为输出到控制台
    :return:日志对象
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

    if stream:
        # 设置CMD日志
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.DEBUG)
        logger.addHandler(sh)


    # 设置文件日志s
    fh = logging.FileHandler(path, encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger


class TrainHistory(keras.callbacks.Callback):
    def __init__(self, log=None, model_name=None, model=None,train_num = 1200):
        super(TrainHistory, self).__init__()
        if not log:
            path = os.path.join(path_log_dir, 'callback.log')
            log = create_log(path=path, stream=False)
        self.log = log
        self.model_name = model_name
        self.epoch = 0
        self.info = []
        self.model = model
        if self.model_name == "IDCNN":
            self.path_model = path_model+'IDCNN/'+str(train_num)
        elif self.model_name == 'BILSTM':
            self.path_model =path_model+'BILSTM/'+str(train_num)
        else:
            self.path_model = path_model+'BERT/'+str(train_num)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        message = f"begin epoch: {self.epoch}"
        self.log.info(message)

    def on_epoch_end(self, epoch, logs={}):
        # message = f'end epoch: {epoch} loss:{logs["loss"]} acc:{logs["accuracy"]} val_loss:{logs["val_loss"]} val_acc:{logs["val_accuracy"]}'
        # self.log.info(message)
        print(logs)
        dicts = {
            'model_name': self.model_name,
            'epoch': self.epoch + 1,
            'loss': logs["loss"],
            'acc': logs['acc'],
            'val_loss': logs["val_loss"],
            'val_acc': logs['val_acc'],
            'auc': logs['auc'],
            'val_auc': logs['val_auc'],
        }
        # self.info.append(dict
        save_path = os.path.join(self.path_model, str(epoch) + ".h5")
        self.model.save(save_path)
        message = f'{self.model_name} epoch: {self.epoch}  loss:{logs["loss"]} '
        self.log.info(message)

    def on_batch_end(self, batch, logs={}):
        dicts = {
            'model_name': self.model_name,
            'epoch': self.epoch + 1,
            'loss': logs["loss"],
            'acc': logs['acc'],
            'val_loss': logs["val_loss"],
            'val_acc': logs['val_acc'],
            'auc': logs['auc'],
            'val_auc': logs['val_auc'],
        }
        message = f'{self.model_name} epoch: {self.epoch} batch:{batch} dict:{dicts} '
        self.log.info(message)
        if batch %15 == 0:
            save_path = os.path.join(self.path_model, str(self.epoch) + "_"+str(batch)+ ".h5")
            self.model.save(save_path)
