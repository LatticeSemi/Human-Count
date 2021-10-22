#!/bin/bash

export GPUID=0
export NET="squeezeDet"
export TRAIN_DIR="./logs/humancnt/"
export IMAGE_SET="train"

export TRAIN_DATA_DIR="/home/shubhamjoshi/Human-Count/gcp_dataset/"
export BUCKET_NAME="softnautics_dataset_bucket"
export BUCKET_DATA_PATH="kitti/"

if [ $# -eq 0 ]
then
  echo "Usage: ./scripts/train.sh [options]"
  echo " "
  echo "options:"
  echo "-h, --help                show brief help"
  echo "-net                      (squeezeDet|squeezeDet+|vgg16|resnet50)"
  echo "-gpu                      gpu id"
  echo "-train_dir                directory for training logs"
  exit 0
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/train.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-net                      (squeezeDet|squeezeDet+|vgg16|resnet50)"
      echo "-gpu                      gpu id"
      echo "-train_dir                directory for training logs"
      exit 0
      ;;
    -net)
      export NET="$2"
      shift
      shift
      ;;
    -gpu)
      export GPUID="$2"
      shift
      shift
      ;;
    -train_dir)
      export TRAIN_DIR="$2"
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

case "$NET" in 
  "squeezeDet")
    export PRETRAINED_MODEL_PATH="./data/SqueezeNet/squeezenet_v1.1.pkl"
    ;;
  "squeezeDet+")
    export PRETRAINED_MODEL_PATH="./data/SqueezeNet/squeezenet_v1.0_SR_0.750.pkl"
    ;;
  "resnet50")
    export PRETRAINED_MODEL_PATH="./data/ResNet/ResNet-50-weights.pkl"
    ;;
  "vgg16")
    export PRETRAINED_MODEL_PATH="./data/VGG16/VGG_ILSVRC_16_layers_weights.pkl"
    ;;
  *)
    echo "net architecture not supported."
    exit 0
    ;;
esac


python3 ./src/train.py \
  --dataset=KITTI \
  --bucket_name=$BUCKET_NAME \
  --pretrained_model_path=$PRETRAINED_MODEL_PATH \
  --data_path=$TRAIN_DATA_DIR \
  --bucket_data_path=$BUCKET_DATA_PATH \
  --gcp=True \
  --image_set=$IMAGE_SET \
  --train_dir="$TRAIN_DIR/train" \
  --net=$NET \
  --max_steps=250000 \
  --summary_step=100 \
  --checkpoint_step=500 \
  --gpu=$GPUID

#  --max_steps=900000 \
