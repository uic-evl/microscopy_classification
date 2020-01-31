from pathlib import Path
from os import listdir
from shutil import copytree, copy, move
from skimage.transform import resize
from skimage.io import imread, imsave
import random
import pandas as pd

def _copy_images(source, dest, classes):
    total_files = 0
    total_copied = 0
    print(source)
    print(dest)
    for c in classes:        
        source_class = source / c
        target_class = dest / c
        imgs = [x for x in listdir(source_class) if (source_class / x).is_file()]
        total_files += len(imgs)
        for img in imgs:
            source_img = source_class / img
            target_img = target_class / img
            try:
                copy(source_img, target_img)
                total_copied += 1
            except Exception as e:
                print(e)
    print("%d/%d images copied" % (total_copied, total_files))

def create_ds1(output_dir, clef_x_path, clef_13_path, prefix='ds1'):
    clef_x_path = Path(clef_x_path)
    clef_13_path = Path(clef_13_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # copy the original image_clef content
    output_name = '%s_%s' % (prefix, clef_x_path.name)
    copytree(clef_x_path, output_dir / output_name)        
    
    classes = [x for x in listdir(clef_x_path / 'train') if (clef_x_path / 'train' / x).is_dir()]
    _copy_images((clef_13_path / 'train'), (output_dir / output_name / 'train'), classes)
      

def create_validation_split(input_path, split=10, seed=1):
    input_path = Path(input_path)
    val_path = input_path / 'validation'
    val_path.mkdir(exist_ok=True)
    
    classes = [x for x in listdir(input_path / 'train') if (input_path / 'train' / x).is_dir()]
    train_paths = [input_path / 'train' / c for c in classes]
    no_imgs_train = [len(listdir(c)) for c in train_paths]
    no_imgs_to_val = [count // split for count in no_imgs_train]
    
    train_data = [listdir(train_path) for train_path in train_paths]
    for d in train_data:
        random.shuffle(d)
    
    val_data = [train_data[i][0:no_imgs_to_val[i]] for i in range(len(classes))]
    train_data = [train_data[i][no_imgs_to_val[i]:] for i in range(len(classes))]
        
    for i in range(len(val_data)):
        (val_path / classes[i]).mkdir(exist_ok=True)
        for img in val_data[i]:
            source = input_path / 'train' / classes[i] / img
            dest = val_path / classes[i] / img
            move(source, dest)
    
    no_imgs_train = [len(listdir(c)) for c in train_paths]
    val_paths = [val_path / c for c in classes]
    no_imgs_val = [len(listdir(c)) for c in val_paths]

    print("Classes:")
    print(classes)
    print("Number of images in train:")
    print(no_imgs_train)
    print("Number of images in validation:")
    print(no_imgs_val)
    

def preprocess_train_val_set(input_path, target_size=(224, 224, 3)):
    # Create a copy of the training and validation set such 
    # that we don't need to use ImageGenerator transformations.
    # Useful for RandomSearch.
    input_path = Path(input_path)
    source_train = input_path / 'train'
    source_validation = input_path / 'validation'    
    classes = [x for x in listdir(source_train) if (source_train / x).is_dir()]
    
    # create a copy of the train and validation images
    target_suffix = '%s_%s' % (target_size[0], target_size[1])
    source_train_resized = input_path / ('train_' + target_suffix)
    source_val_resized = input_path / ('validation_' + target_suffix) 
    copytree(source_train, source_train_resized) 
    copytree(source_validation, source_val_resized) 

    for c in classes:
        _resize_images(source_train_resized / c, target_size)          
        _resize_images(source_val_resized / c, target_size)          


def _resize_images(source_folder, target_size):
    img_names = [x for x in listdir(source_folder)]
    imgs = [imread(source_folder / name) for name in img_names]
    resized_imgs = [(resize(img, target_size) * 255).astype('uint8') for img in imgs]
    for name, img in zip(img_names, resized_imgs):
        imsave(source_folder / name, img)
        

def create_test_2016(clef16_path, output_dir):
    clef16_path = Path(clef16_path)
    output_dir = Path(output_dir)
    (output_dir / 'test').mkdir(exist_ok=True)

    df = pd.read_csv(clef16_path / 'SubfigureClassificationTest2016GT.csv', sep = ' ')
    df.columns = ['name', 'class']
    classes = df['class'].unique()
    total_files = len(listdir(clef16_path / 'test'))

    for c in classes:
        (output_dir / 'test' / c).mkdir(exist_ok=True)

    copied = 0
    for idx, row in df.iterrows():
        img_name = row['name'] + '.jpg'
        source = clef16_path / 'test' / img_name
        dest = output_dir / 'test' / row['class'] / img_name
        try:
            copy(source, dest)
            copied += 1
        except Exception as e:
            print(row['name'])
            print(e)
    
    print("copied %d/%d" % (copied, total_files))
    return
