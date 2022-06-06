import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import autokeras as ak
from ak_custom_layers import *
import argparse
import os
import sys

def train(data_dir="",log_dir="",total_train_images=0,IMG_SHAPE=32,BATCH_SIZE=128,color_mode="grayscale",input_channels=1,quantrelu=True,kernel_quant=True,separable=False,\
          use_batchnorm=True,dropout=0.8,epochs=50,max_trials=20,boundaries=[],tuner="hyperband",max_model_size=1600000,over_write=False,callbacks=[]) :
    print(data_dir,log_dir,total_train_images,IMG_SHAPE,BATCH_SIZE,color_mode,input_channels,quantrelu,kernel_quant,separable,\
          use_batchnorm,dropout,epochs,max_trials,boundaries,tuner,max_model_size,over_write,callbacks)
    #sys.exit()
    train_ds = ak.image_dataset_from_directory(directory=data_dir, #os.path.join(data_dir,"train"),
                                                   seed=195,
                                                   shuffle=True,
                                                   validation_split=0.2,
                                                   subset="training",
                                                   color_mode=color_mode,
                                                   image_size=(IMG_SHAPE, IMG_SHAPE),
                                                   batch_size=BATCH_SIZE)

    val_ds = ak.image_dataset_from_directory(directory=data_dir, #os.path.join(data_dir,"val"),
                                                   seed=195,
                                                   shuffle=True,
                                                   validation_split=0.2,
                                                   subset="validation",
                                                   color_mode=color_mode,
                                                   image_size=(IMG_SHAPE, IMG_SHAPE),
                                                   batch_size=BATCH_SIZE)

    #class_names = train_ds.class_names
    #print(class_names)

    print("VALUES BEFORE SCALING")
    image_batch, labels_batch = next(iter(train_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))
    
    
    train_ds = train_ds.map(lambda x, y: (x*0.0078125, y))
    print("VALUES AFTER SCALING")
    image_batch, labels_batch = next(iter(train_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))
    val_ds = val_ds.map(lambda x, y: (x*0.0078125, y))
    
    #AUTOTUNE = tf.data.AUTOTUNE
    #train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    #val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay( boundaries=boundaries, values=[0.1,0.01,0.001,0.0001,0.00001])
    cust_opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    kwargs = {"optimizer":cust_opt,"metrics":["accuracy"]}
    input_node = ak.Input(shape=(IMG_SHAPE,IMG_SHAPE,input_channels), name="img")
    output_node1 = LatticeConvBlock(separable=separable, use_batchnorm=use_batchnorm,quantrelu=quantrelu,kernel_quant=kernel_quant,img_size=IMG_SHAPE)(input_node)#(mid_node)
    output_node2 = LatClassificationHead(kernel_quant=kernel_quant,dropout=dropout)(output_node1)
    auto_model = ak.AutoModel(inputs=input_node, outputs=output_node2, overwrite=over_write, max_trials=max_trials,tuner=tuner,\
            seed=99, max_model_size=max_model_size,objective="val_loss",**kwargs)
    

    # Search
    auto_model.fit(x=train_ds,epochs=epochs,verbose=1,validation_data=val_ds,callbacks=callbacks)
    auto_model_test_loss, auto_model_test_acc = auto_model.evaluate(val_ds,verbose=2)
    print("best model eval data loss and accuracy : {},{} ".format(auto_model_test_loss, auto_model_test_acc))
    best_model = auto_model.export_model()
    fc_index = 0
    for i in range(-10,0,1):
        #print(i,net_model.get_layer(index=i).name)
        if best_model.get_layer(index=i).name=="dense":
            #print(net_model.get_layer(index=i))
            fc_index = i
            print("Found the dense layer index {}".format(fc_index))
    if fc_index==0:
        print("No dense layer index found. Please check the name of the dense layer in this script")
    intermediate_model = tf.keras.Model(best_model.input,best_model.get_layer(index=fc_index).output)
    intermediate_model.save("./logs/intrm_model_loss_{}_acc_{}.h5".format(auto_model_test_loss, auto_model_test_acc),save_format="h5")
    best_model.save("./logs/best_model.h5",save_format="h5")
    print(best_model.summary())
    print("Please note that with image_dataset_from_directory of AutoKeras, the order of classes will be [1,10,11,2,3,4,5,6,7,8,9] instead of [1,2,3,4,5,6,7,8,9,10,11]")
    return

def main(args):
    handgesture = True
    if handgesture:
        data_dir = args.data_dir
        total_train_images = args.total_train_images

        train_dir = data_dir+"/train"
        val_dir=data_dir+"/val"
        IMG_SHAPE = args.image_resolution  # Image HxW
        input_channels = 1  # 1 or 3
        color_mode = args.color_mode
        CLASS_NAMES = None # ak.image_from_directory does not take class names. So by default order of classes will be
        # instead of [1,2,3,4,5,6,7,8,9,10,11] it will be [1,10,11,2,3,4,5,6,7,8,9]


    BATCH_SIZE = 128 #800000 # batch size of images to be pre-processed in train and/or val dataset. Keep it at a value > #total images
    # The actual batch size will be inside fit method. Defaulted to 32
    epochs = args.epochs #50
    max_trials = args.max_trials # Max number of architectures to be tried by AK to train the best one
    max_model_size=1600000 # To limit the model size
    dropout = 0.8 # Dropout rate

    separable = args.separable # True for Mobilenet like model
    use_batchnorm = True
    kernel_quant = args.kernel_quant # whether to use kernel quantization -0.5<=W<0.5
    quantrelu = args.quantrelu # Whether to provide activation quantization
    tuner = "hyperband"  # Optimizer used by AK to find the best set of hyperparameters

    steps_per_epoch = math.ceil(total_train_images/BATCH_SIZE)
    total_steps = math.ceil(steps_per_epoch*epochs)
    lt1 = math.ceil(0.65*total_steps)
    lt2 = math.ceil(0.85*total_steps)
    lt3 = math.ceil(0.95*total_steps)
    lt4 = total_steps
    boundaries = [lt1, lt2, lt3, lt4]
    print("steps per epoch {}".format(steps_per_epoch))
    print("total steps {}".format(total_steps))
    print("four limits for SGD are {} , {} , {} and {}".format(lt1,lt2,lt3,lt4))
    print("four limits for SGD are in epochs {} , {} , {} and {}".format(lt1//steps_per_epoch,lt2//steps_per_epoch,lt3//steps_per_epoch,lt4//steps_per_epoch))

    model_folder_extn = data_dir.split("/")[-1] if data_dir[-1]!="/" else data_dir.split("/")[-2]
    model_dir = "./logs/"+model_folder_extn+"/" # directory where models will be stored
    model_name = "ak_saved_model" # model saved by autokeras directly
    ckpt_model_1 = model_dir+"ckpt_files/" # Similar to ckpt_model and this will be used as the final model

    mckpt1 = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_model_1, monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=False, mode='min', save_freq='epoch')
    callbacks = [mckpt1]
    # Train the model
    train(data_dir,args.log_dir,total_train_images,IMG_SHAPE,BATCH_SIZE,color_mode,input_channels,quantrelu,kernel_quant,separable,\
          use_batchnorm,dropout,epochs,max_trials,boundaries,tuner,max_model_size,args.over_write,callbacks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, required=True, help="Input dir of images")
    parser.add_argument("-l", "--log_dir", type=str, required=True, help="Log directory to store AutoKeras models and the final model")
    parser.add_argument("-t", "--total_train_images", type=int, required=True, help="Total number of training images. Approximately 80% of total augmented images")
    parser.add_argument("-i", "--image_resolution", type=int, default=32, help="Input resolution")
    parser.add_argument("-c", "--color_mode", type=str, default="grayscale", help="Grayscale or rgb")
    parser.add_argument("-qr", "--quantrelu", type=bool, default=True, help="Activation Quantization")
    parser.add_argument("-kq", "--kernel_quant", type=bool, default=True, help="Kernel Quantization")
    parser.add_argument("-s", "--separable", type=bool, default=False, help="True for Mobilenet structure")
    parser.add_argument("-ep", "--epochs", type=int, default=50, help="Total number of epochs used")
    parser.add_argument("-mxt", "--max_trials", type=int, default=7, help="Total number of structures for AutoKeras to try and select the best one from")
    parser.add_argument("-o", "--over_write", type=bool, default=False, help="whether to overwrite the previous training")
    args = parser.parse_args()

    main(args)
