''' Create the train, validation and test splits
    from the ImageClef dataset for microscopy
    classes (DMEL, DMTR, DMLI, DMFL)
'''

from pathlib import Path
from shutil import copytree, move
from os import listdir
from argparse import ArgumentParser
import random
import math


def main():
    parser = ArgumentParser()
    parser.add_argument('-clef', type=str, help='path to imageclef files')
    parser.add_argument('-out', type=str, help='where to store the files')
    parser.add_argument('-seed', type=int, default=50, help='random seed')
    parser.add_argument('-split', type=int, default=10, help='train - validation split')
    args = parser.parse_args()

    if args.clef == '':
        raise Exception('Invalid path provided for clef')
    if args.out == '':
        raise Exception('Invalid path provided for out')

    CLEF = Path(args.clef)
    OUTPUT = Path(args.out)
    CLASSES = ['DMEL', 'DMFL', 'DMLI', 'DMTR']
    SETS = ['train', 'validation', 'test']
    SPLIT = args.split
    SEED = args.seed

    # create structure
    OUTPUT.mkdir(exist_ok=True)

    for c in CLASSES:
        source = CLEF / 'train' / c
        dest = OUTPUT / 'train' / c
        copytree(source, dest)

    train_paths = [OUTPUT / 'train' / c for c in CLASSES]
    no_imgs_train = [len(listdir(c)) for c in train_paths]
    SPLIT = SPLIT / 100
    no_imgs_to_val = [math.ceil(count * SPLIT) for count in no_imgs_train]

    # shuffle and move to validation folder
    random.seed(SEED)
    # list of images organized by class in CLASSES
    train_data = [listdir(CLEF / 'train' / c) for c in CLASSES]
    for d in train_data:
        random.shuffle(d)   # in-place shuffling

    val_data = [train_data[i][0:no_imgs_to_val[i]] for i in range(len(CLASSES))]
    train_data = [train_data[i][no_imgs_to_val[i]:] for i in range(len(CLASSES))]

    (OUTPUT / 'validation').mkdir(exist_ok=True)
    for i in range(len(val_data)):
        (OUTPUT / 'validation' / CLASSES[i]).mkdir(exist_ok=True)
        for img in val_data[i]:
            source = OUTPUT / 'train' / CLASSES[i] / img
            dest = OUTPUT / 'validation' / CLASSES[i] / img
            move(source, dest)

    no_imgs_train = [len(listdir(c)) for c in train_paths]
    val_paths = [OUTPUT / 'validation' / c for c in CLASSES]
    no_imgs_val = [len(listdir(c)) for c in val_paths]

    print("Classes:")
    print(CLASSES)
    print("Number of images in train:")
    print(no_imgs_train)
    print("Number of images in validation:")
    print(no_imgs_val)


if __name__ == "__main__":
    main()
