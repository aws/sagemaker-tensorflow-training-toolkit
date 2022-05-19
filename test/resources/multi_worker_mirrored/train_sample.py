import tensorflow as tf
import numpy as np


strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(2, input_shape=(5,)),
        ]
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)


def dataset_fn(ctx):
    x = np.random.random((2, 5)).astype(np.float32)
    y = np.random.randint(2, size=(2, 1))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return dataset.repeat().batch(1, drop_remainder=True)


dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)

model.compile()
model.fit(dist_dataset)
