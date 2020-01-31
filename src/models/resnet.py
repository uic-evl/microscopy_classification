from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adamax, Adam

####
# Possible output layers:
# conv2_block1_out, conv2_block2_out
# conv3_block1_out, conv3_block2_out, conv3_block3_out, conv3_block4_out
# conv4_block1_out, conv4_block2_out, conv4_block3_out, conv4_block4_out, conv4_block5_out, conv4_block6_out
# conv5_block1_out, conv5_block2_out, conv5_block3_out
####

def build_resnet(num_classes=30, output_layer='conv5_block3_out', fine_tune=None, lr=2e-3, optimizer='adamax'):
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    output_layer = resnet.get_layer(output_layer).output
            
    trainable = False
    if not fine_tune:
        fine_tune = "########"
    elif fine_tune == 'all':
        trainable = True

    for layer in resnet.layers:
        if fine_tune == layer.name:
            trainable = True
        layer.trainable = trainable

    avg_pooling = GlobalAveragePooling2D()(output_layer)
    softmax = Dense(num_classes, activation='relu', kernel_initializer='glorot_uniform')(avg_pooling)
    model = Model(inputs=resnet.input, outputs=softmax)

    if optimizer == 'adamax':
        optimizer = Adamax(learning_rate=lr)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=lr)
    
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

