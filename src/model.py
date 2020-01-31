import numpy as np
import datetime
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnet
from tensorflow.keras.applications import VGG16, VGG19, InceptionResNetV2
from tensorflow.keras.applications import ResNet50, Xception, ResNet101
from tensorflow.keras.applications import ResNet152, InceptionV3, NASNetLarge
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as KerasModel
from pathlib import Path

_VGG16 = 'VGG16'
_VGG19 = 'VGG19'
_INCEPTION_RESNET_V2 = 'inception_resnet_v2'
_RESNET50 = 'resnet50'
_RESNET101 = 'resnet101'
_INCEPTION_V3 = 'inception_v3'
_XCEPTION = 'xception'
_NASNET = 'nasnet_large'


class Model():
    def __init__(self, cfg):
        self.train_dir = cfg.train_dir
        self.validation_dir = cfg.validation_dir
        self.test_dir = cfg.test_dir
        self.input_shape = Model.get_input_shape(cfg.backbone)
        self.backbone_name = cfg.backbone
        
        if cfg.saved_model_path is not None and cfg.output_layer is not None:
            # whenever we have a fine-tuned models
            self.backbone = Model.load_saved_models(cfg.saved_model_path, cfg.output_layer)
        else:                        
            self.backbone = Model.load_backbone(cfg.backbone, self.input_shape)
        self.preprocess_input = Model.get_preprocess_input(cfg.backbone)
        self.use_augmentation = cfg.use_augmentation
        self.dropout = cfg.dropout
        self.lr = cfg.lr
        self.batch_size = cfg.batch_size
        self.epochs = cfg.epochs
        self.tensorboard_path = Path(cfg.tensorboard)
        self.ds = Path(self.train_dir).parent.name

    def train(self):
        # create model based on required operation.
        # for feature extraction, not much is required
        model = self.backbone

        train_datagen = Model.get_train_datagen(self.use_augmentation,
                                                self.preprocess_input)
        test_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.input_shape[0], self.input_shape[0]),
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True)

        # what if i don't have 20 images
        validation_generator = test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(self.input_shape[0], self.input_shape[0]),
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=True)
        return

    #!TODO validate for augmentation
    def extract_features(self, directory, sample_count, no_classes,
                         aug=False, mult=10, logdir=None, suffix=''):
        shape = self.backbone.output.shape  # tensor shape [None, x, y ,z]

        # if there is augmentation, we will iterate mult times the
        # dataset to produce more samples
        if aug:
            sample_count = int(sample_count * mult)
            datagen = ImageDataGenerator(
                preprocessing_function=self.preprocess_input,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input)

        filenames = []
        features = np.zeros(shape=(sample_count, shape[1], shape[2], shape[3]))
        labels = np.zeros(shape=(sample_count, no_classes))

        # in case we want to get features for specific elements
        if sample_count < self.batch_size:
            iter_batch_size = sample_count
        else:
            iter_batch_size = self.batch_size

        generator = datagen.flow_from_directory(
            directory,
            target_size=(self.input_shape[0], self.input_shape[0]),
            batch_size=iter_batch_size,
            class_mode='categorical',
            shuffle=True)

        # i, j = 0, 0
        # generator return all elements in epoch before
        # extracting from the next one. Takes into account
        # a last batch with less elements than batch_size
        start = 0
        for input_batch, labels_batch in generator:
            # guard against smaller last batches on each epoch
            local_batch_size = input_batch.shape[0]
            end = start + local_batch_size

            features_batch = self.backbone.predict(input_batch)
            # generator.index_array contains the order of the shuffled ids
            filenames += [generator.filenames[x] for x in generator.index_array[start: end]]
            features[start: end] = features_batch
            labels[start: end] = labels_batch
            # i += 1
            # j += 1

            start = end
            # reset the local counter after each epoch
            # if j * iter_batch_size >= len(generator.filenames):
            #     j = 0
            # check the global counter against the required amount
            if start >= sample_count:
                break
        if logdir:
            # save the features and labels as a npz file
            use_aug = "noaug"
            if aug:
                use_aug = "aug"
            output_name = ("%s_%s_%s_%s.npz") % \
                          (self.ds, self.backbone_name, use_aug, suffix)
            output_name = logdir / output_name
            np.savez_compressed(output_name, features, labels, filenames)
        return features, labels, filenames

    @staticmethod
    def get_callbacks(log_dir):
        tensorboard = TensorBoard(logdir=log_dir, write_graph=True,
                                  write_images=True)
        csv_logger = CSVLogger(log_dir / 'logger.csv', append=True)
        best_checkpoint = ModelCheckpoint(filepath=log_dir / 'best.hdf5',
                                          monitor='val_acc',
                                          verbose=1,
                                          save_best_only=True)
        chk_file_path = log_dir / 'weights.{epoch:02d}.hdf5'
        checkpoint = ModelCheckpoint(filepath=chk_file_path,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=False,
                                     period=50)
        return [tensorboard, csv_logger, best_checkpoint, checkpoint]

    @staticmethod
    def get_input_shape(backbone):
        if backbone == _VGG16 or backbone == _VGG19:
            return (224, 224, 3)
        elif backbone == _INCEPTION_RESNET_V2 or backbone == _INCEPTION_V3:
            return (299, 299, 3)
        elif backbone == _RESNET50 or backbone == _RESNET101:
            return (224, 224, 3)
        elif backbone == _NASNET:
            return (331, 331, 3)
        elif backbone == _XCEPTION:
            return (299, 299, 3)
    
    @staticmethod
    def load_saved_models(model_dir, output_layer):    
        my_model = load_model(model_dir)
        conv_base = my_model.get_layer(output_layer).output
        saved_model = KerasModel(inputs=my_model.input, outputs=conv_base)
        return saved_model
    
    
    @staticmethod
    def load_backbone(backbone, input_shape):
        conv_base = None
        if backbone == _VGG16:
            conv_base = VGG16(weights='imagenet',
                              include_top=False,
                              input_shape=input_shape)
        if backbone == _VGG19:
            conv_base = VGG19(weights='imagenet',
                              include_top=False,
                              input_shape=input_shape)
        elif backbone == _INCEPTION_RESNET_V2:
            conv_base = InceptionResNetV2(weights='imagenet',
                                          include_top=False,
                                          input_shape=input_shape)
        elif backbone == _INCEPTION_V3:
            conv_base = InceptionV3(weights='imagenet',
                                    include_top=False,
                                    input_shape=input_shape)
        elif backbone == _RESNET50:
            conv_base = ResNet50(weights='imagenet',
                                 include_top=False,
                                 input_shape=input_shape)
        elif backbone == _RESNET101:
            conv_base = ResNet101(weights='imagenet',
                                  include_top=False,
                                  input_shape=input_shape)
        elif backbone == _XCEPTION:
            conv_base = Xception(weights='imagenet',
                                 include_top=False,
                                 input_shape=input_shape)
        elif backbone == _NASNET:
            conv_base = NASNetLarge(weights='imagenet',
                                    include_top=False,
                                    input_shape=input_shape)
        return conv_base

    @staticmethod
    def get_preprocess_input(backbone):
        preprocess_input = None
        if backbone == _VGG16 or backbone == _VGG19:
            preprocess_input = preprocess_input_vgg
        elif backbone == _INCEPTION_RESNET_V2:
            preprocess_input = preprocess_input_inception_resnet_v2
        elif backbone == _RESNET50 or backbone == _RESNET101:
            preprocess_input = preprocess_input_resnet
        elif backbone == _INCEPTION_V3:
            preprocess_input = preprocess_input_inception_v3
        elif backbone == _XCEPTION:
            preprocess_input = preprocess_input_xception
        elif backbone == _NASNET:
            preprocess_input = preprocess_input_nasnet
        return preprocess_input

    @staticmethod
    def get_train_datagen(use_augmentation, preprocess_input):
        if use_augmentation:
            return ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                channel_shift_range=10,
                horizontal_flip=True,
                fill_mode='nearest')
        return ImageDataGenerator(preprocessing_function=preprocess_input)
