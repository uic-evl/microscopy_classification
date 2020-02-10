# Microscopy Modality Classification

Repository for the modality classification subtask of microscopy images. The microscopy classes include electron microscopy (DMEL), light microscopy (DMLI), transmission microscopy (DMTR) and fluorescence microscopy (DMFL) based on the taxonomy from the [ImageClef Medical Task 2016](https://www.imageclef.org/2016/medical).

## Approaches
* Feature extraction and SVMs: In this approach, we extracted the image features from the microscopy dataset using a ResNet model. Then, we used those features to train a Support Vector Machine model. An optional intermediate step consisted in applying dimensionality reduction with PCA.
* Deep Learning with Fine Tuning: End-to-end approach for classification using ResNet50, shallower version of ResNet50, or a shallow CNN-6 (Yu et al.)

## Data
The ImageCLEF2016 and ImageCLEF2013 datasets are available upon request to the ImageCLEF organizers.

The main requirement is to have the data organized in the following format:

```
DATASET
    train
        class_1
        class_2
        ...
        class_n
    validation
        class_1
        class_2
        ...
        class_n
    test
        class_1
        class_2
        ...
        class_n        
```

Given a training set with the following organization:
```
imageclef_16
    class_1
    class_2
    ...
    class_n
```
We used *data.py* to split the data into a training and validation set in a stratified manner. We used additional utility functions in *utils/data.py* to create the *ds1* dataset (name borrowed from Yu et. al) that contains the ImageCLEF16 and ImageCLEF13 images. Our test set is the ImageCLEF16 test partition.


## Train and Test SVMs
The process consists of two three steps. First, get the features from the feature extractor (backbone) and save the training, validation and test features as *.npz* files. Second, train the SVM model and optionally apply PCA. Third, try the model on the test set.

```
extract_features.py 
    -backbone BACKBONE_MODEL
    -input_dir PATH_TO_DATASET
    -output_dir OUTPUT
    -batch_size 32
    -suffix SUFFIX
    -classes CLASSES_TO_CONSIDER
    -output_layer OUTPUT_LAYER
    -fine_tuned_dir FINE_TUNED_DIR

BACKBONE_MODEL: resnet50 or other flavors from the pre-trained Keras models
PATH_TO_DATASET: Folder containing the images organized in folder classes.
OUTPUT: Output folder. The script creates an output folder inside. We use the output folder name as the dataset name.
SUFFIX: train, validation or test. Used to distinguish the outputs.
CLASSES: "DMEL,DMFL,DMLI,DMTR" by default
OUTPUT_LAYER: '' if we want the features from the last layer. An specific layer output name (depends on the model) if we want the features from a specific layer.
FINE_TUNED_DIR: Fine-tuned model path if we don't want to use the default ImageNet weigths. Used with OUTPUT_LAYER.

Output: A .npz file with the features.
```

```
train_svm.py
    -path FEATURES_FOLDER
    -ds DATASET
    -bb FEATURE_EXTRACTOR
    -out OUTPUT_FOLDER
    -w USE_WEIGHTS
    -k KERNEL
    -pca USE_PCA
    -n_components N_COMPONENTS
    -aug false    

FEATURES_FOLDER: Folder containing the features.
DATASET: Dataset name or name of the folder containing the features.
FEATURE_EXTRACTOR: resnet50 or model used in the extract_features script.
OUTPUT_FOLDER: Where to save the model and output values.
USE_WEIGHTS: false (default). Use weights for SVM due to unbalanced set or not.
KERNEL: linear or rbf
USE_PCA: false (default). Apply PCA before training data?
N_COMPONENTS: Number of components to use in PCA. If it's a value between 0 and 1, represents the percentage of variance explained. If it's an integer, it is a given number of components.

Output: model.pkl file, output values with parameters and validation results, confusion matrix for validation and set of misclassified images.
```

The current implementation needs to use the training features to fit the PCA function as we are not saving that model dimensionality reduction model on the training step. This is only relevant when *-pca true* is used.
```
test_svm.py
    -feats_path FEATURES_FOLDER
    -model_path MODEL_PATH
    -ds DATASET
    -bb FEATURE_EXTRACTOR
    -pca USE_PCA
    -n_components N_COMPONENTS

FEATURES_FOLDER: Same as in train_svm.py
MODEL_PATH: Folder containing the trained model.pkl file
DATASET: Same as in train_svm.py
FEATURE_EXTRACTOR: Same as in train_svm.py
USE_PCA: Same as in train_svm.py
N_COMPONENTS: Same as in train_svm.py

Output: Classification report on test set.
```

## Train and Test Deep Learning Models

```
train.py
    -model_name resnet50, cnn6 or vgg16
    -run_name Name for output folder
    -output_dir Artifacts folde path
    -batch_size 32
    -epochs Number of epochs (default 1000)
    -lr Learning rate (default 1e-3)
    -patience Patience for early stopping (default 15)
    -dataset Name of folder containing the training and validation folders
    -datapath Folder containing the dataset folder
    -fine_tune Name of last non-trainable layer (default: '', just train head)
    -optimizer adam or adamax (default: adam)
    -use_decay Use learning rate decay every 5 steps (default: 0 for false)
    -num_classes Number of classification classes (default: 30)
    -use_default_preprocessing Use Keras preprocessing function or custom one (default: 0 for false)
    -resnet_output Output layer for shallower ResNet versions (default: conv5_block3 for full model)
    -vgg_output Output layer for shallower VGG versions (default: block5_pool for full model)

Output: Best model based on training loss (best.hdf5), best model based on validation loss (best_val.hdf5), and last trained model (last.hdf5), logger.csv with accuracy and loss values per epoch.
```

```
test.py
    -model_name resnet50, cnn6 or vgg16
    -model_path Path to .hdf5 weights
    -hdf5 Specific model to load (default: best.hdf5)
    -test_path Path to test files (e.g. ../dataset/test)

Output: Classification report for test set.

```



# Contact
Juan Trelles jtrell2@uic.edu 

