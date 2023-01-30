import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class TimeSeriesClassifier:
    def __init__(self, input_shape, num_classes):
        self.model = self._build_model(input_shape, num_classes)

    def _build_model(self, input_shape, num_classes):
        """
        Builds the Convolutional Neural Network (CNN) model.
        :param input_shape: Tuple (num_timesteps, num_features)
        :param num_classes: Integer, number of classes
        :return: Keras model instance
        """
        # Define input layer
        input_layer = layers.Input(shape=input_shape)

        # Define 1D convolution layer with 32 filters and a kernel size of 3
        conv1 = layers.Conv1D(32, 3, activation='relu')(input_layer)

        # Define max pooling layer with a pool size of 2
        max_pool1 = layers.MaxPooling1D(2)(conv1)

        # Define another 1D convolution layer with 64 filters and a kernel size of 3
        conv2 = layers.Conv1D(64, 3, activation='relu')(max_pool1)

        # Define max pooling layer with a pool size of 2
        max_pool2 = layers.MaxPooling1D(2)(conv2)

        # Flatten the output of the max pooling layer
        flatten = layers.Flatten()(max_pool2)

        # Define a dense layer with 128 neurons and a 'relu' activation function
        dense1 = layers.Dense(128, activation='relu')(flatten)

        # Define output layer with 'num_classes' neurons and a 'softmax' activation function
        output_layer = layers.Dense(num_classes, activation='softmax')(dense1)

        # Build the model using the defined layers
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        # Compile the model with categorical crossentropy loss and 'adam' optimizer
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def train(self, X, y, epochs, batch_size):
        """
        Trains the model using the provided training data.
        :param X: 3D numpy array of shape (num_samples, num_timesteps, num_features)
        :param y: 2D numpy array of shape (num_samples, num_classes)
        :param epochs: Integer, number of training epochs
        :param batch_size: Integer, size of the training batch
        """
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
