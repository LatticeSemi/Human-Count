import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import autokeras as ak
import numpy as np
import binary_ops
from binary_ops import *
import argparse
import sys

BATCH_SIZE = 128

def evaluate(args):
    print("From the given dataset only 20% data is used for evaluation.")
    image_gen_train = ImageDataGenerator(rescale=1./128,validation_split=0.2)
    test_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                         directory=args.data_dir,
                                                         shuffle=True,
                                                         subset="validation",
                                                         color_mode=args.color_mode,
                                                         target_size=(args.image_resolution,args.image_resolution),
                                                         classes=['1','10','11','2','3','4','5','6','7','8','9'],
                                                         seed = 195
                                                         )

    cust_obj1 = { "bo": binary_ops,"binary_ops": binary_ops,"lin_8b_quant": lin_8b_quant,\
                                                                        "FixedDropout" : FixedDropout,\
                                                                       "MyInitializer": MyInitializer,\
                                                                       "MyRegularizer": MyRegularizer,\
                                                                       "MyConstraints": MyConstraints,\
                                                                       "CastToFloat32":CastToFloat32}

    net_model = tf.keras.models.load_model(args.model,custom_objects=cust_obj1)

    loss,metric = net_model.evaluate(x=test_data_gen)
    print(loss,metric)
    #yprob = net_model.predict(x=test_data_gen)
    #yp = yprob.argmax(axis=-1)
    #print(yp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="./data/augmented_dataset", help="Input dir of images")
    parser.add_argument("-l", "--model", type=str, default="./logs/best_model.h5", help="Log directory to store AutoKeras models and the final model")
    parser.add_argument("-i", "--image_resolution", type=int, default=32, help="Input resolution")
    parser.add_argument("-c", "--color_mode", type=str, default="grayscale", help="Grayscale or rgb")
    args = parser.parse_args()

    evaluate(args)
