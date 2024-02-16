import keras
from keras import layers

input_shape = (765,572,3)
kernel_size = (3, 3)

# Import Dataset
training_ds = keras.utils.image_dataset_from_directory(
    directory  = 'resources/PH2/training_data',
    labels     = 'inferred',
    label_mode ='categorical',
    image_size = (765, 572),
    batch_size = 16
)
testing_ds = keras.utils.image_dataset_from_directory(
    directory  = 'resources/PH2/testing_data',
    labels     = 'inferred',
    label_mode ='categorical',
    image_size = (765, 572),
    batch_size = 16
)

# VGG-16 Instantiation
model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=kernel_size, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax') 
])

# Model Compilation
model.compile(
    optimizer = keras.optimizers.Adam(lr=0.001), # learning rate
    loss      = keras.losses.categorical_crossentropy,
    metrics   = ['accuracy']
)

checkpoint = keras.callbacks.ModelCheckpoint( # monitor accuracy
    'vgg16_1.h5',
    monitor           = 'val_accuracy', 
    verbose           = 1, # type of progress display
    save_best_only    = True,
    save_weights_only = False,
    mode              = 'auto',
    period            = 1
)

early = keras.callbacks.EarlyStopping( # early stopping if there is no improvement
    monitor   = 'val_accuracy',
    min_delta = 0,
    patience  = 20, # early stop if no improvement in 20 epochs
    verbose   = 1,
    mode      = 'auto'
)

hist = model.fit_generator(
    steps_per_epoch  = 10,
    generator        = training_ds,
    validation_data  = testing_ds,
    validation_steps = 10,
    epochs           = 20,
    callbacks        = [checkpoint, early]
)

