# Starter code from https://keras.io/guides/training_with_built_in_methods/
import keras
from keras import layers

# =============================================================================
# Hyperparams & Input Data
# =============================================================================
num_classes = 10
input_shape = (32, 32, 3)

epochs = 60
train_batch_size = 128
val_batch_size = 256

# =============================================================================
# Import data
# Documentation: https://keras.io/api/datasets/cifar10/
# =============================================================================
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

# Reserve 5,000 samples for validation
x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]

# =============================================================================
# Build model
# =============================================================================
model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# =============================================================================
# Train model
# =============================================================================
print("Train model")
history = model.fit(
    x_train,
    y_train,
    batch_size=train_batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
)

# =============================================================================
# Evaluate model
# =============================================================================
print("Test model")
results = model.evaluate(x_test, y_test, batch_size=val_batch_size)

# print("Generate predictions for 3 samples")
# predictions = model.predict(x_test[:3])
# print("predictions shape:", predictions.shape)

# # =============================================================================
# # Save model
# # =============================================================================
# model.save('cifar10_convnet_model.keras')
# print('Saved model!')