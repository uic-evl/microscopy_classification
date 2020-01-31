import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from os import listdir

BATCH_SIZE = 32

def main():
    parser = ArgumentParser()
    parser.add_argument('-model_name', type=str, default='cnn6', help='cnn6, resnet50 or vgg16')
    parser.add_argument('-model_path', type=str, default='', help='folder storing model hdf5 weights')
    parser.add_argument('-hdf5', type=str, default='best.hdf5', help='model to load')
    parser.add_argument('-test_path', type=str, default='', help='path to test files')
    
    args = parser.parse_args()
    
    path = Path(args.model_path) / args.hdf5
    test_path = Path(args.test_path)
    model = load_model(path)
    
    # get the number of images in the dataset
    total_images  = 0
    folders = [f for f in listdir(test_path) if (test_path / f).is_dir()]    
    for f in folders:
        total_images += len(listdir(test_path / f))
    print('Found %d images ' % total_images)
    
    if args.model_name == 'cnn6':
        test_datagen = ImageDataGenerator(rescale=1./255)
        target_size = (224, 224)
    elif args.model_name == 'resnet50':
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_resnet)
        target_size = (224, 224)
    elif args.model_name == 'vgg16':
        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
        target_size = (224, 224)
        
    
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=target_size,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False)
    
    # put the items from one epoch into numpy arrays
    # don't want to use evaluate_generator to be able to calculate f1 too
    count = 0
    X, Y = [], []
    
    while count < total_images:
        batch = test_generator.next()
        for image in batch[0]:
            X.append(image)
        for label in batch[1]:
            Y.append(label)
        count += batch[1].shape[0]
    X = np.array(X)
    Y = np.array(Y)
    
    results = model.evaluate(X, Y, batch_size=32)
    print('test loss, test acc:', results)
    
    y_probs = model.predict(X)    
    y_pred = np.argmax(y_probs, axis=1)
    y_pred_categorical = to_categorical(y_pred)
        
    print(classification_report(Y, y_pred_categorical, digits=4, target_names=test_generator.class_indices))


if __name__ == '__main__':
    main()
