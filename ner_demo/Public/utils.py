import logging
from tensorflow import saved_model,keras
import os
from Public.path import save_model_path, log_path, model_version_info
from logs.logger import Logging


class TrainHistory(keras.callbacks.Callback):
    def __init__(self, model_name=None, class_model=None, train_num=1200):
        super(TrainHistory, self).__init__()

        self.log = Logging(log_path + 'train_' + model_version_info + '.log').create_logging()

        self.model_name = model_name
        self.epoch = 0
        self.info = []
        self.class_model = class_model
        self.save_model_path = save_model_path + model_version_info + '/'

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        message = f"begin epoch: {self.epoch}"
        self.log.info(message)

    def on_epoch_end(self, epoch, logs={}):
        dicts = {
            'model_name': self.model_name,
            'epoch': self.epoch + 1,
            'loss': logs.get("loss"),
            'acc': logs.get('acc'),
            'val_loss': logs.get("val_loss"),
            'val_acc': logs.get('val_acc'),
            'auc': logs.get('auc'),
            'val_auc': logs.get('val_auc'),
        }

        message = f'{self.model_name} , epoch: {self.epoch} , dict:{dicts} '
        self.log.info(message)
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        model_weight_path = os.path.join(self.save_model_path, str(epoch) + '_epoch.weight')

        self.class_model.save(model_weight_path)
        # saved_model(self.model,save_path)


    # def on_batch_end(self, batch, logs={}):
    #     print(logs)
    #     dicts = {
    #         'model_name': self.model_name,
    #         'epoch': self.epoch + 1,
    #         'loss': logs.get("loss"),
    #         'acc': logs.get('acc'),
    #         'val_loss': logs.get("val_loss"),
    #         'val_acc': logs.get('val_acc'),
    #         'auc': logs.get('auc'),
    #         'val_auc': logs.get('val_auc'),
    #     }
    #     message = f'{self.model_name} epoch: {self.epoch} batch:{batch} dict:{dicts} '
    #     self.log.info(message)
    #     if batch % 15 == 0 and batch :
    #
    #         save_path = os.path.join(self.path_model, str(self.epoch) + "_"+str(batch)+ ".weight")
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
    #         self.model.save_weights(save_path)
