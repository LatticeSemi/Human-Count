from easydict import EasyDict


def get_config():
    cfg = EasyDict()

    cfg.dataset_path = "/home/softnautics/datasets/face-recognition-datasets/ms-celebs-1m/"
    cfg.validation_set = "/home/softnautics/datasets/face-recognition-datasets/validation-set/"
    cfg.test_set = "/home/softnautics/datasets/face-recognition-datasets/test-set/"

    cfg.MODEL_NAME = 'Face_Identification'
    cfg.LOG_PATH = "logs"
    cfg.BACK_BONE = "MV1"

    cfg.IMAGE_WIDTH = 112
    cfg.IMAGE_HEIGHT = 112
    cfg.N_CHANNELS = 1

    cfg.init = None  # "<Pretrained H5 file path>"

    cfg.FILTER_DEPTHS = [40, 40, 60, 60, 80, 80, 100, 40]
    cfg.EARLY_POOLING = True
    cfg.USE_CONV3 = True

    cfg.EPOCHS = 300
    cfg.QUANT_RANGE = (0, 2)

    cfg.FEATURES = 128
    cfg.BATCHSIZE = 256

    cfg.GPUID = 0
    cfg.LEARNING_RATE = 0.01
    cfg.REDUCELRONPLATEAU = True

    cfg.TEST_BATCHSIZE = 64
    cfg.IS_CROP = False

    return cfg
