import datetime
from pathlib import Path
from argparse import ArgumentParser
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger, LearningRateScheduler
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cnn6 import build_cnn6
from models.resnet import build_resnet
from models.vgg16 import build_vgg16

def main():
    parser = ArgumentParser()
    parser.add_argument('-model_name', type=str, default='cnn6', help='cnn6, resnet50 or vgg16')
    parser.add_argument('-run_name', type=str, default='test', help='log dir prefix')
    parser.add_argument('-output_dir', type=str, default='/mnt/clef/artifacts/', help='where to save outputs')    
    parser.add_argument('-batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-epochs', type=int, default=1000, help='epochs')
    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-patience', type=int, default=15, help='early stopping patience')    
    parser.add_argument('-dataset', type=str, default='ds1_imageclef_2016', help='dataset used for training')
    parser.add_argument('-datapath', type=str, default='/mnt/clef', help='source folder for datasets')
    parser.add_argument('-resnet_output', type=str, default='conv5_block3_out', help='activation layer output')
    parser.add_argument('-vgg_output', type=str, default='block5_pool', help='vgg activation layer output')
    parser.add_argument('-fine_tune', type=str, default='', help='min trainable layer in backbone')    
    parser.add_argument('-avoid_validation', type=int, default=0, help='1 for just using training data')
    parser.add_argument('-num_classes', type=int, default=30, help='imageclef classes')
    parser.add_argument('-optimizer', type=str, default='adam', help='adamax or adam')
    parser.add_argument('-use_decay', type=int, default=0, help='use learning rate decay: 1, no: 0')
    parser.add_argument('-use_default_preprocessing', type=int, default=1, help='use preprocessing function provided by keras or custom one')
    
    args = parser.parse_args()
    
    avoid_validation = args.avoid_validation == 1
    use_decay = args.use_decay == 1
    use_default_preprocessing = args.use_default_preprocessing == 1

    out_dir = Path(args.output_dir)    
    train_dir = Path(args.datapath) / args.dataset / 'train'
    val_dir = Path(args.datapath) / args.dataset / 'validation'

    fine_tune = args.fine_tune
    if args.fine_tune == '':
        fine_tune = None
        
    if args.model_name == 'cnn6':
        model = build_cnn6(num_classes=args.num_classes, lr=args.lr, n_fcs=1, n_neurons=512, optimizer=args.optimizer)
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        target_size = (224, 224)
    elif args.model_name == 'resnet50':
        model = build_resnet(num_classes=args.num_classes, output_layer=args.resnet_output, fine_tune=fine_tune, lr=args.lr, optimizer=args.optimizer)
        if use_default_preprocessing:
            preprocessing_function = preprocess_input_resnet
        else:
            preprocessing_function = preprocess_resnet_clef
        train_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        val_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
        target_size = (224, 224) 
    elif args.model_name == 'vgg16':
        model = build_vgg16(num_classes=args.num_classes, output_layer=args.vgg_output, fine_tune=fine_tune, lr=args.lr)
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
        target_size = (224, 224) 

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        class_mode='categorical',
        batch_size=args.batch_size,
        shuffle=True)

    if avoid_validation:
        validation_generator = None
        validation_steps = None
    else:
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=target_size,
            class_mode='categorical',
            batch_size=args.batch_size,
            shuffle=True)
        validation_steps = validation_generator.samples // args.batch_size

    steps_per_epoch = train_generator.samples // args.batch_size
    
    
    return train(model, args.run_name, out_dir, train_generator, validation_generator, 
                 steps_per_epoch, validation_steps, epochs=args.epochs, patience=args.patience, use_decay=use_decay)


def train(model, run_name, out_dir, train_generator, val_generator, steps_per_epoch, validation_steps, epochs=100, patience=15, 
          use_decay=False):
    now = datetime.datetime.now().strftime("%Y%m%y_%H%M%S")    
    run_name = "%s_%s" % (run_name, now)
    out_dir = Path(out_dir / run_name)
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / 'summary.txt','w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
    callbacks = get_callbacks(out_dir, patience=patience, use_decay=use_decay)
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks)    
    model.save(out_dir / 'last.hdf5')
    return history

def step_decay(epoch, lr):
    decay_rate = 0.5
    decay_step = 5
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def get_callbacks(log_dir, patience=15, use_decay=False):
    early_stopping = EarlyStopping(patience=patience)
    tensorboard = TensorBoard(log_dir=log_dir, write_graph=True)
    best_checkpoint = ModelCheckpoint(str(log_dir / 'best.hdf5'), monitor='loss', save_best_only=True)
    best_val_checkpoint = ModelCheckpoint(str(log_dir / 'best_val.hdf5'), monitor='val_loss', save_best_only=True)
    chk_file_path = str(log_dir / 'weights.{epoch:02d}.hdf5')
    checkpoint = ModelCheckpoint(filepath=chk_file_path,
                                    monitor='val_acc',
                                    verbose=1,
                                    save_best_only=False,
                                    period=10)
    csv_logger = CSVLogger(str(log_dir / 'logger.csv'), append=True)
    lr_scheduler = LearningRateScheduler(step_decay, verbose=1)
    
    # removed checkpoint temporarily: lack of storage
    if use_decay:
        return [tensorboard, best_checkpoint, best_val_checkpoint, csv_logger, early_stopping, lr_scheduler]
    else:
        return [tensorboard, best_checkpoint, best_val_checkpoint, csv_logger, early_stopping]

def preprocess_resnet_clef(x, **kwargs):
    backend = kwargs.get('backend', None)
    data_format = backend.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))
    
    x /= 255.
    
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
    return x
    
    
if __name__ == '__main__':
    main()
    
    
    #https://github.com/keras-team/keras-applications/blob/976050c468ff949bcbd9b9cf64fe1d5c81db3f3a/keras_applications/imagenet_utils.py#L18
    
    #http://cs231n.github.io/neural-networks-3/#sanitycheck