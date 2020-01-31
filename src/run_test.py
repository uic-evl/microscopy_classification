from models.cnn6 import build_cnn6
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def main():
    cnn6 = build_cnn6()
    train_dir = '/mnt/clef/ds1_imageclef_2016/train'
    val_dir = '/mnt/clef/ds1_imageclef_2016/validation'
    batch_size=32
    
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)
    
    steps_per_epoch = 7709 // batch_size
    validation_steps = 842 // batch_size
    
    cnn6.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    history = cnn6.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_steps)
    
if __name__ == '__main__':
    main()