from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import os

train_data_dir = 'data/train/'
validation_data_dir = 'data/val/'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
target_size = (48, 48)
color_mode = 'grayscale'

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=target_size,
    color_mode=color_mode,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=target_size,
    color_mode=color_mode,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),

    Conv2D(256, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.1),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

num_train_imgs = sum([len(files) for _, _, files in os.walk(train_data_dir)])
num_test_imgs = sum([len(files) for _, _, files in os.walk(validation_data_dir)])

print("Number of training images:", num_train_imgs)
print("Number of test images:", num_test_imgs)

epochs = 30
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_imgs // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // batch_size)

model.save('model_file.h5')
