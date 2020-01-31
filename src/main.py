import yaml
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to folder containing train, validation and test images')
    parser.add_argument('output_path', type=str, help='path used to save the results')
    parser.add_argument('mode', type=str, help='run mode, train or test')
    parser.add_argument('backbone', type=str, help='backbone cnn model: vgg16, vgg19, resnet50')
    parser.add_argument('action', type=str, help='fine tuning or image extraction')
    parser.add_argument('lr', type=float, help='learning rate')

if __name__ == '__main__':
    main()
