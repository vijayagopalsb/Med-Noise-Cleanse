# File: src/core/model.py

# Implement the Denoising Autoencoder Model

# Import Libraries

# Add this BEFORE importing tensorflow
import os
import sys
import absl.logging

# Suppress all types messages tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress absl warnings
absl.logging.set_verbosity(absl.logging.ERROR)

# Add project root to sys.path to resolve module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers # type: ignore
from keras.saving import register_keras_serializable # type: ignore
from config.logging_config import logger

# Disable GPU if not needed
tf.config.set_visible_devices([], 'GPU')
logger.info("Disableing GPU !!!")
logger.info("Tensorflow Warnings suppressed successfully!")

@register_keras_serializable()
def hybrid_loss(y_true, y_pred):
    # mse = tf.reduce_mean(tf.square(y_true - y_pred))
    # ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    # return 0.5 * mse + 0.5 * ssim
    # mse = tf.reduce_mean(tf.square(y_true - y_pred))
    # ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    # return 0.7 * mse + 0.3 * ssim  # favor brightness
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.8 * mse + 0.2 * ssim


class DenoisingAutoencoder:

    def __init__(self, input_shape=(256, 256, 1)):
        self.input_shape = input_shape
        self.model = self._create_model()

    def _create_model(self):
        return self._build_deep_autoencoder()


    def summary(self):
        """Prints model summary."""
        self.model.summary()

    def get_model(self):
        """Return the compiled model."""
        return self.model

    def _build_model(self):
        """Create the denoising autoencoder model."""
        inputs = layers.Input(shape=self.input_shape)

        # Encoder with skip connections
        e1 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
        e1 = layers.BatchNormalization()(e1)
        p1 = layers.MaxPooling2D((2, 2))(e1)

        e2 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
        e2 = layers.BatchNormalization()(e2)
        p2 = layers.MaxPooling2D((2, 2))(e2)

        # Bottleneck and dropout
        b = layers.Conv2D(256, (3,3), activation="relu", padding="same")(p2)
        b = layers.Dropout(0.5)(b)

        # Decoder with skip connections
        d1 = layers.Conv2DTranspose(128, (3, 3), strides =2, activation="relu", padding="same")(b)
        d1 = layers.concatenate([d1, e2])
        d1 = layers.BatchNormalization()(d1)

        d2 = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(d1)
        d2 = layers.concatenate([d2, e1])
        outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d2)

        #x = layers.Conv2D(32, (3, 3), activation="relu", padding="same") (inputs)
        #x = layers.MaxPooling2D((2, 2), padding = "same")(x)
        #x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        #x = layers.MaxPooling2D((2, 2), padding= "same")(x)

        # Bottleneck with dropout
        #x = layers.Conv2D(128, (3, 3), activation="relu", padding = "same")(x)

        # Decoder
        #x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
        #x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
        #outputs = layers.Conv2D(1, (3,3), activation="sigmoid", padding="same")(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model


    # from tensorflow.keras import layers, models

    def _build_deep_autoencoder(self):
        inputs = layers.Input(shape=self.input_shape)

        # ----------- Encoder -----------
        e1 = layers.Conv2D(64, (3, 3), padding="same")(inputs)
        e1 = layers.LeakyReLU(negative_slope=0.1)(e1)
        e1 = layers.BatchNormalization()(e1)
        p1 = layers.MaxPooling2D((2, 2), padding="same")(e1)

        e2 = layers.Conv2D(128, (3, 3), padding="same")(p1)
        e2 = layers.LeakyReLU(negative_slope=0.1)(e2)
        e2 = layers.BatchNormalization()(e2)
        p2 = layers.MaxPooling2D((2, 2), padding="same")(e2)

        # ----------- Bottleneck -----------
        b = layers.Conv2D(256, (3, 3), padding="same")(p2)
        b = layers.LeakyReLU(negative_slope=0.1)(b)
        # Dropout skipped for now

        # ----------- Decoder -----------
        d1 = layers.Conv2DTranspose(128, (3, 3), strides=2, padding="same")(b)
        d1 = layers.LeakyReLU(negative_slope=0.1)(d1)
        d1 = layers.Concatenate()([d1, e2])
        d1 = layers.Conv2D(128, (3, 3), padding="same")(d1)
        d1 = layers.LeakyReLU(negative_slope=0.1)(d1)

        d2 = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same")(d1)
        d2 = layers.LeakyReLU(negative_slope=0.1)(d2)
        d2 = layers.Concatenate()([d2, e1])
        d2 = layers.Conv2D(64, (3, 3), padding="same")(d2)
        d2 = layers.LeakyReLU(negative_slope=0.1)(d2)

        outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(d2)

        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

        # model.compile(optimizer="adam", loss="mse")
        # model.compile(optimizer="adam", loss=hybrid_loss)


        # return models.Model(input_img, decoded)


