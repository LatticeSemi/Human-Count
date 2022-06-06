1. To augment the given dataset run :
python3 ./data/augmentation_canvas.py --input_dir=./data/original_handgesture_dataset --output_dir=./data/augmented_dataset

2. To begin the training with autokeras run :
python3 ak_training.py --data_dir=./data/augmented_dataset --log_dir=./logs --total_train_images=755357

3. To evaluate the best model run :
python3 ak_evaluate.py
