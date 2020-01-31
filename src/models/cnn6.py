import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPool2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax, Adam

def build_cnn6(num_classes=30, lr=2e-3, n_fcs=1, n_neurons=512, optimizer='adamax'):
        
    inputs = Input(shape=(224, 224, 3), name='inputs')
    # block 1
    conv2d_1 = Conv2D(32, (3,3), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='conv2d_1')(inputs)
    # block 2
    conv2d_2 = Conv2D(32, (3,3), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='conv2d_2')(conv2d_1)
    maxpool_2 = MaxPool2D(pool_size=(2, 2), name='maxpool2d_2')(conv2d_2)
    dropout_2 = Dropout(0.25)(maxpool_2)
    # block 3
    conv2d_3 = Conv2D(64, (3,3), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='conv2d_3')(dropout_2)
    # block 4
    conv2d_4 = Conv2D(64, (3,3), activation='relu', kernel_initializer='glorot_uniform', padding='same', name='conv2d_4')(conv2d_3)
    maxpool_4 = MaxPool2D(pool_size=(2, 2), name='maxpool2d_4')(conv2d_4)
    dropout_4 = Dropout(0.25)(maxpool_4)
    # block 5, optionally adding more FCs
    flatten = Flatten(name='flatten_5')(dropout_4) # replace this one by global average pooling

    prev = flatten
    for i in range(n_fcs):
        fc_5 = Dense(n_neurons, activation='relu', kernel_initializer='glorot_uniform')(prev)
        dropout_5 = Dropout(0.5)(fc_5)
        prev = dropout_5

    # output layer
    softmax = Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform')(prev)
    
    model = Model(inputs=inputs, outputs=softmax)
    
    if optimizer == 'adamax':
        optimizer = Adamax(learning_rate=lr)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=lr)
        
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    
    return model
    