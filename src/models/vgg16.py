from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax, Adam

####
# UPDATE! Possible output layers:
# conv2_block1_out, conv2_block2_out
# conv3_block1_out, conv3_block2_out, conv3_block3_out, conv3_block4_out
# conv4_block1_out, conv4_block2_out, conv4_block3_out, conv4_block4_out, conv4_block5_out, conv4_block6_out
# conv5_block1_out, conv5_block2_out, conv5_block3_out
####

def build_vgg16(num_classes=30, output_layer='block5_pool', fine_tune=None, lr=2e-3, optimizer='adamax'):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    output_layer = vgg.get_layer(output_layer).output
    
    trainable = False
    if not fine_tune:
        fine_tune = "########"

    for layer in vgg.layers:
        layer.trainable = trainable
        if fine_tune == layer.name:
            trainable = True

    flatten = Flatten()(output_layer)
    fc1 = Dense(4096, activation='relu', kernel_initializer='glorot_uniform')(flatten)
    fc2 = Dense(4096, activation='relu', kernel_initializer='glorot_uniform')(fc1)
    softmax = Dense(num_classes, activation='relu', kernel_initializer='glorot_uniform')(fc2)
    model = Model(inputs=vgg.input, outputs=softmax)

    if optimizer == 'adamax':
        optimizer = Adamax(learning_rate=lr)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=lr)    
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model
