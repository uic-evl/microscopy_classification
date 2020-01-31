from model import Model
from argparse import ArgumentParser
from os import listdir
from pathlib import Path
import numpy as np

class Cfg():
    def __init__(self, bb):
        self.train_dir = '/home/jtrell2/Documents/tensorboard'
        self.validation_dir = '/home/jtrell2/Documents/tensorboard'
        self.test_dir = '/home/jtrell2/Documents/tensorboard'
        self.backbone = bb
        self.dropout = True
        self.lr = 1e-4
        self.batch_size = 32
        self.epochs = 100
        self.tensorboard = '/home/jtrell2/Documents/tensorboard'
        self.use_augmentation = True

def main():
    parser = ArgumentParser()
    parser.add_argument('-backbone', type=str, default='resnet50',
                        help='VGG16, VGG19, inception_resnet_v2, resnet50, resnet101, inception_v3, xception, nasnet_large')
    parser.add_argument('-input_dir', type=str, help='path to folder with folder classes')
    parser.add_argument('-output_dir', type=str, help='where to save outputs')
    parser.add_argument('-augmentation', type=int, default=0, help='use image augmentation')
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-classes', type=str, default='DMEL,DMFL,DMLI,DMTR', help='modality classes')
    parser.add_argument('-suffix', type=str, default='train')
    parser.add_argument('-fine_tuned_dir', type=str, default='')
    parser.add_argument('-output_layer', type=str, default='')
    
    args = parser.parse_args()
    
    augmentation = True
    if args.augmentation == 0:
        augmentation = False
        
    fine_tuned_dir = None
    if args.fine_tuned_dir != '':
        fine_tuned_dir = args.fine_tuned_dir
    
    output_layer = None
    if args.output_layer != '':
        output_layer = args.output_layer
    
        
    cfg = Cfg(args.backbone)
    cfg.use_augmentation = augmentation
    cfg.saved_model_path = fine_tuned_dir
    cfg.output_layer = output_layer
    
    model = Model(cfg)
    
    classes = args.classes.split(',')
    paths = [Path(args.input_dir) / c for c in classes]
    sample_count = [len(listdir(x)) for x in paths]
    total = np.sum(sample_count)
    print(sample_count)
    print(total)
        
    features, labels, filenames = \
        model.extract_features(args.input_dir, total, len(classes), 
                           logdir=Path(args.output_dir), suffix=args.suffix,
                           aug=augmentation)

if __name__ == '__main__':
    main()