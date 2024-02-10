import warnings
warnings.filterwarnings('ignore') 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import hw2_complete as hw
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization, Dense, Flatten, MaxPooling2D, Add, Activation, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

def load_and_prepare_data():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    train_images, val_images, train_labels, val_labels = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)

    return (train_images, train_labels), (x_test, y_test), (val_images, val_labels)

def build_model1():
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    for _ in range(4):
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model2():
    inputs = Input(shape=(32, 32, 3))
    
    x = SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)

    x = Conv2D(64, (1, 1), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for _ in range(4):
        x = Conv2D(64, (1, 1))(x)
        x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def build_model3():
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(105, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for _ in range(3):
        residual = x
        x = Conv2D(105, (3, 3), padding='same')(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(105, (3, 3), padding='same')(x)  
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Add()([x, residual])  

    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = Flatten()(x)
    x = Dense(248, activation='relu')(x)  
    x = BatchNormalization()(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model50k():
    inputs = Input(shape=(32, 32, 3))
    
    # Initial Convolutional Layer
    x = Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    for filters in [32, 64]:
        if filters > 32:  
            x = Dropout(0.2)(x)
        
        # Convolutional Block
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        if filters == 64:  
            shortcut = Conv2D(filters, (1, 1), padding='same', kernel_regularizer=l2(1e-4))(inputs)  
            x = Add()([x, shortcut])
        
    x = Flatten()(x)
    
    # Final Dense Layer
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)  
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def load_or_build_model50k():
    try:
        model50k = tf.keras.models.load_model("best_model.h5")
        print("Model50k loaded successfully.")
    except Exception as e:  
        print(f"Error loading model from 'best_model.h5': {e}. Building model50k from scratch.")
        model50k = build_model50k()
    return model50k

model50k = load_or_build_model50k()

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels), (val_images, val_labels) = load_and_prepare_data()
    
    model1 = build_model1()
    model1.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels), batch_size=64)
    model1.summary() 

    model2 = build_model2()
    model2.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels), batch_size=64)
    model2.summary()  

    model3 = build_model3()
    model3.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels), batch_size=64)
    model3.summary()  

    model50k = build_model50k()
    model50k.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels), batch_size=64)
    model50k.summary()  
    model50k.save("best_model.h5") 