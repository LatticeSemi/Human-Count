import numpy as np
import tensorflow.keras as keras
from glob import glob
import cv2
import albumentations as A
import cv2


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, directory, batch_size=512, dim=(112, 112), n_channels=1, shuffle=True, gen_type='train', num_person_per_epoch=10000):
        """Initialization"""
        self.gen_type = gen_type
        self.num_person_per_epoch = num_person_per_epoch
        self.paths = glob(directory + "**/")
        self.paths1 = []
        for path in self.paths[:self.num_person_per_epoch]:
            self.paths1.extend(glob(path + "**"))
        self.paths = self.paths1
        self.classes = np.unique(np.asarray([path.split('/')[-2] for path in self.paths]))
        self.index_mapping = {}
        for i, class_id in enumerate(self.classes):
            self.index_mapping[class_id] = i
        self.n_classes = len(self.classes)
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        # Declare an augmentation pipeline
        self.transform = A.Compose([
            A.RandomBrightnessContrast(0.2, 0.2, False, p=0.2),
            A.Rotate(limit=(-15,15), interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0,
                     mask_value=0, always_apply=False, p=0.3)

        ])
        self.on_epoch_end()
        print("Total ", len(self.paths), " with ", self.n_classes, " number of classes found.")

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int((len(self.paths) / self.batch_size))

    def __augment(self, image):
        transformed_image = self.transform(image=image)["image"]
        return transformed_image

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = indexes
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        import random
        random.shuffle(self.paths)
        for path in self.paths[:self.num_person_per_epoch]:
            self.paths1.extend(glob(path + "**"))
        self.paths = self.paths1
        self.classes = np.unique(np.asarray([path.split('/')[-2] for path in self.paths]))
        self.index_mapping = {}
        for i, class_id in enumerate(self.classes):
            self.index_mapping[class_id] = i
        self.n_classes = len(self.classes)
        self.indexes = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.__len__()

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = cv2.imread(self.paths[ID])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, self.dim)
            img = self.__augment(img)
            img = img[:, :, None]
            img = img / 128.
            X[i, ] = img
            # Store class
            y[i] = self.index_mapping[self.paths[ID].split('/')[-2]]
        return X, y
