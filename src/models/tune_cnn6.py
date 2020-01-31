import numpy as np

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from cnn6 import build_cnn6
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = '/mnt/clef/ds1_imageclef_2016/train'
val_path = '/mnt/clef/ds1_imageclef_2016/validation'
total_train = 7709
total_val = 842
batch_size = 32


def main():
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(224, 224),
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True)

    steps_train = (total_train // batch_size) + 1
    steps_val = (total_val // batch_size) + 1

    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for i in range(steps_train):
        xs, ys = train_generator.next()
        for j in range(xs.shape[0]):
            x_train.append(xs[j,:])
            y_train.append(ys[j,:])
            
    for i in range(steps_val):
        xs, ys = val_generator.next()
        for j in range(xs.shape[0]):
            x_val.append(xs[j,:])
            y_val.append(ys[j,:])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    cnn6_reg = KerasRegressor(build_cnn6)
    param_distribs = {
        "lr": reciprocal(5e-4, 1e-2),
        "n_neurons": [128, 256, 512],
        "n_fcs": [1, 2, 3]
    }

    early_stopping = EarlyStopping(patience=10)

    rnd_search_cv = RandomizedSearchCV(cnn6_reg, param_distribs, n_iter=10, cv=3, n_jobs=-1)
    rnd_search_cv.fit(x_train, y_train, epochs=100,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping])

if __name__ == '__main__':
    main()