"""Three convolution blocks: a compact custom CNN that is easy to explain."""
import tensorflow as tf

from .config import CLASSES, TrainConfig


def build_model(config: TrainConfig):
    layers = tf.keras.layers
    inputs = layers.Input(shape=(config.image_size, config.image_size, 1), name="grayscale_pixels")
    x = layers.Rescaling(1 / 255.0, name="pixel_scaling")(inputs)
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=config.seed),
        layers.RandomRotation(config.rotation, fill_mode="reflect", seed=config.seed + 1),
        layers.RandomTranslation(config.translation, config.translation, fill_mode="reflect", seed=config.seed + 2),
        layers.RandomZoom((-config.zoom, config.zoom), fill_mode="reflect", seed=config.seed + 3),
    ], name="training_only_augmentation")
    x = augmentation(x)  # Keras activates these random layers only when training=True.
    for i, width in enumerate(config.filters, 1):
        x = layers.Conv2D(width, 3, padding="same", use_bias=False, name=f"conv_{i}")(x)
        x = layers.BatchNormalization(name=f"batch_norm_{i}")(x)
        x = layers.ReLU(name=f"relu_{i}")(x)
        if i < len(config.filters):
            x = layers.MaxPooling2D(name=f"pool_{i}")(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dense(config.dense_units, activation="relu", name="dense_features")(x)
    x = layers.Dropout(config.dropout, seed=config.seed, name="dropout")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inputs, outputs, name="steel_defect_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
