"""
    Author      : Abid Indraswara
    Date Create : 11 December 2024
    Modul Training Model Data
    Fungsi :
    - Modul untuk menjalankan training model berdasarkan
      arsitektur yang sudah disusun.
"""

# Import Library Umum
import os

# Import Tensorflow
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import tensorflow_transform as tft

# Import Beberapa Fungsi & Variabel dari Modul Transform
from depression_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)

# Funsgi Menjalankan Model sesuai Arsitektur
def get_model(show_summary=True):
    """
    This function defines a Keras model and returns the model as a
    Keras object.
    """

    # Transform One-Hot Fitur Kategori
    input_features = []
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=transformed_name(key))
        )

    # Transform One-Hot Fitur Numerik
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    # Arsitektur Model
    concatenate = tf.keras.layers.concatenate(input_features)
    deep = tf.keras.layers.Dense(256, activation="relu")(concatenate)
    deep = tf.keras.layers.Dense(64, activation="relu")(deep)
    deep = tf.keras.layers.Dense(16, activation="relu")(deep)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)

    # Menyimpan Model ke Variabel
    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    
    # Kompilasi Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    # Menampilkan Ringkasan Model
    if show_summary:
        model.summary()

    return model

# Fungsi Load Dataset Tensorflow yang terkompress dalam format GZIP
def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# Fungsi untuk konversi dan memproses data ke dalam model serving Tensorflow
def get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn

# Fungsi untuk prepare dataset pelatihan dan setup model
def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Generates features and labels for tuning/training.
    Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of
        returned dataset to combine in a single batch
    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset

# Fungsi Training dan Save Model
def run_fn(fn_args):
    """Train the model based on given args.
    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)

    model = get_model()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=5
    )

    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )

    plot_model(
        model,
        to_file='images/model_plot.png',
        show_shapes=True,
        show_layer_names=True
    )