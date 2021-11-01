import argparse
import os

from absl import logging

from model import binary_ops
from model.binary_ops import *
from model.evaluations import get_val_data, perform_val
from config.create_config import *


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


def main(arguments):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(arguments.gpuid)

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    model_path = arguments.model
    model = tf.keras.models.load_model(model_path, compile=False,
                                       custom_objects={"bo": binary_ops,
                                                       "binary_ops": binary_ops,
                                                       "lin_8b_quant": lin_8b_quant,
                                                       "MyInitializer": MyInitializer,
                                                       "MyRegularizer": MyRegularizer,
                                                       "MyConstraints": MyConstraints,
                                                       "tf": tf})
    # print(model.summary())
    configuration = get_config()
    cfg = {
        "test_dataset": configuration.test_set,
        "embd_shape": configuration.FEATURES,
        "batch_size": configuration.TEST_BATCHSIZE,
        "is_ccrop": configuration.IS_CROP
    }

    print("[*] Loading LFW ") # , AgeDB30 and CFP-FP...")
    lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
        get_val_data(cfg['test_dataset'])

    print("[*] Perform Evaluation on LFW...")
    acc_lfw, best_th = perform_val(
        cfg['embd_shape'], cfg['batch_size'], model, lfw, lfw_issame,
        is_ccrop=cfg['is_ccrop'])
    print("    acc {:.4f}, th: {:.2f}".format(acc_lfw, best_th))

    # print("[*] Perform Evaluation on AgeDB30...")
    # acc_agedb30, best_th = perform_val(
    #     cfg['embd_shape'], cfg['batch_size'], model, agedb_30,
    #     agedb_30_issame, is_ccrop=cfg['is_ccrop'])
    # print("    acc {:.4f}, th: {:.2f}".format(acc_agedb30, best_th))

    # print("[*] Perform Evaluation on CFP-FP...")
    # acc_cfp_fp, best_th = perform_val(
    #     cfg['embd_shape'], cfg['batch_size'], model, cfp_fp, cfp_fp_issame,
    #     is_ccrop=cfg['is_ccrop'])
    # print("    acc {:.4f}, th: {:.2f}".format(acc_cfp_fp, best_th))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test FaceID model.')
    parser.add_argument("--gpuid", default=-1, help="GPU id to run testing")
    parser.add_argument("--model", default=None, help="Model Path")
    args = parser.parse_args()
    main(args)
