import keras
from keras import layers

input_shape = (765,572,3)
kernel_size = (5, 5)

# Import Dataset
training_ds = keras.utils.image_dataset_from_directory(
    directory  = 'resources/PH2/training_data',
    labels     = 'inferred',
    label_mode ='categorical',
    image_size =(765, 572)
)
testing_ds = keras.utils.image_dataset_from_directory(
    directory  = 'resources/PH2/testing_data',
    labels     = 'inferred',
    label_mode ='categorical',
    image_size =(765, 572)
)

# VGG-16 Instantiation
model = keras.Sequential([
    layers.Conv2D(input_shape=input_shape, filters=64, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.Conv2D(filters=64, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
    layers.Conv2D(filters=128, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.Conv2D(filters=128, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
    layers.Conv2D(filters=256, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.Conv2D(filters=256, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.Conv2D(filters=256, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
    layers.Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
    layers.Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.Conv2D(filters=512, kernel_size=kernel_size, padding='same', activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
    layers.Flatten(),
    layers.Dense(units=4096, activation='relu'),
    layers.Dense(units=4096, activation='relu'),
    layers.Dense(units=3, activation='softmax')
])

# Model Compilation
model.compile(
    optimizer = keras.optimizers.Adam(lr=0.001), # learning rate
    loss      = keras.losses.categorical_crossentropy,
    metrics   = ['accuracy']
)

checkpoint = keras.callbacks.ModelCheckpoint( # monitor accuracy
    'vgg16_1.h5',
    monitor           = 'val_acc', 
    verbose           = 1, # type of progres display
    save_best_only    = True,
    save_weights_only = False,
    mode              = 'auto',
    period            = 1
)

early = keras.callbacks.EarlyStopping( # early stopping if there is no improvement
    monitor   = 'val_acc',
    min_delta = 0,
    patience  = 20, # early stop if no improvement in 20 epochs
    verbose   = 1,
    mode      = 'auto'
)

hist = model.fit_generator(
    steps_per_epoch = 100,
    generator = training_ds,
    validation_data = testing_ds,
    validation_steps = 10,
    epochs = 100,
    callbacks = [checkpoint, early]
)

