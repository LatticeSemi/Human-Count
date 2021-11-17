import os

import tensorflow_addons as tfa
from absl import logging
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.optimizer_v2.adam import *

from config.create_config import *
from model.data_generator import DataGenerator
from model.model import *
import argparse         


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                logging.info(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)
set_memory_growth()


class Training:
    def __init__(self,cfg):
        self.cfg = cfg

        self.checkpoint_dir = os.path.join(self.cfg.LOG_PATH, "checkpoints")
        self.tb_dir = os.path.join(self.cfg.LOG_PATH, "tensorboard")

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cfg.GPUID)

        self.face_rec = FaceRecognition(self.cfg)
        self.face_rec_model = self.face_rec.model
        self.face_rec_model.compile(optimizer=Adam(learning_rate=self.cfg.LEARNING_RATE),
                                    loss=tfa.losses.TripletSemiHardLoss())
        print(self.face_rec_model.summary())

        self.tripplet_data_generator = DataGenerator(self.cfg.dataset_path,self.cfg.validation_split,batch_size=self.cfg.BATCHSIZE,dim=(self.cfg.IMAGE_HEIGHT,self.cfg.IMAGE_WIDTH),gen_type='train')
        self.val_data_generator = DataGenerator(self.cfg.dataset_path,self.cfg.validation_split,batch_size=self.cfg.BATCHSIZE,dim=(self.cfg.IMAGE_HEIGHT,self.cfg.IMAGE_WIDTH),gen_type='val')
        self.init_epoch = 0
        self.create_new_log_dir_structure()
        self.callbacks = []
        self.add_callbacks()
        self.train_model()
        self.save_model()

    @staticmethod
    def ckpt_sorting(s):
        return int(str(s).split('-')[0].split('.')[1])

    def create_new_log_dir_structure(self):
        # delete old checkpoints and tensorboard stuff
        if tf.io.gfile.exists(self.checkpoint_dir):
            print('Target directory already Exists : {}'.format(self.checkpoint_dir))
        else:
            tf.io.gfile.makedirs(self.checkpoint_dir)
        if tf.io.gfile.exists(self.tb_dir):
            print('Target directory already Exists : {}'.format(self.tb_dir))
        else:
            tf.io.gfile.makedirs(self.tb_dir)

    def add_callbacks(self):
        tbCallBack = TensorBoard(log_dir=self.tb_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.callbacks.append(tbCallBack)

        if self.cfg.REDUCELRONPLATEAU:
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, verbose=1,
                                          patience=5, min_lr=0.000001)
            self.callbacks.append(reduce_lr)

        # add a checkpoint saver
        ckp_saver = ModelCheckpoint(self.checkpoint_dir + "/model.{epoch:02d}-{loss:.2f}.hdf5", monitor='loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=True, mode='auto', period=1)
        self.callbacks.append(ckp_saver)

    def check_for_existing_checkpoints(self):
        if self.cfg.init is not None:
            print("Weights initialized by name from {}".format(self.cfg.init))
            self.face_rec.model.load_weights(self.cfg.init)
            # initial_epoch = int(os.path.basename(self.init_file).split('-')[0].split('.')[1])
        elif len(os.listdir(self.checkpoint_dir)) > 0:
            ckpt_list = os.listdir(self.checkpoint_dir)
            sorted_list = sorted(ckpt_list, key=self.ckpt_sorting)
            ckpt_path = os.path.join(self.checkpoint_dir, sorted_list[-1])
            self.face_rec_model.load_weights(ckpt_path)
            print("Weights initialized by name from {}".format(ckpt_path))
            self.init_epoch = self.ckpt_sorting(sorted_list[-1])

    def train_model(self):
        self.check_for_existing_checkpoints()
        self.save_model()
        self.face_rec_model.fit_generator(self.tripplet_data_generator, epochs=self.cfg.EPOCHS,
                                          validation_data=(self.val_data_generator),
                                          verbose=1, initial_epoch=self.init_epoch, callbacks=self.callbacks)

    def freeze_model(self):
        self.check_for_existing_checkpoints()
        self.save_model()

    def save_model(self):
        tf.keras.models.save_model(
            self.face_rec_model,
            os.path.join(self.cfg.LOG_PATH, f"{self.cfg.MODEL_NAME}.h5"),
            overwrite=True, include_optimizer=True, signatures=None, save_format='h5')
        frozen_model = tf.keras.Model(inputs=[self.face_rec_model.input],
                                      outputs=[self.face_rec_model.get_layer("E_dense").output])
        tf.keras.models.save_model(
            frozen_model,
            os.path.join(self.cfg.LOG_PATH, f"{self.cfg.MODEL_NAME}_Frozen.h5"),
            overwrite=True, include_optimizer=False, signatures=None, save_format='h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keras FaceId training')

    parser.add_argument('--dataset_path',type=str,help="Dataset Path",required=True)
    parser.add_argument('--epochs',required=False,type=int,default=300,help="Number of Epochs")
    parser.add_argument('--num_features',required=False,type=int,default=256,help="Number of embedding features")
    parser.add_argument('--batch_size',required=False,type=int,default=256,help="Batch Size")
    parser.add_argument('--validation_split',required=False,type=int,default=20,help="Validation split in percentage")
    parser.add_argument('--logdir',type=str,default='logs/',help="Checkpoint Save Directory",required=True)
    parser.add_argument('--gpu',type=int,default=0,help="Checkpoint Save Directory",required=True)
    args = parser.parse_args()
    cfg = get_config()
    cfg.dataset_path= args.dataset_path
    cfg.EPOCHS= args.epochs
    cfg.FEATURES = args.num_features
    cfg.BATCHSIZE=args.batch_size
    cfg.validation_split = args.validation_split
    cfg.LOG_PATH=args.logdir
    cfg.GPUID=args.gpu
    
    Training(cfg)
