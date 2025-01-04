
# Cat vs Dog Classifier
<p align="center">
  <img src="images/ddog-vs-cat-cover-photo.jpeg" width="600" height="600"/>
</p>


This repository contains a comprehensive implementation of a **Cat vs Dog Classifier** using Convolutional Neural Networks (CNNs) built with TensorFlow and Keras. The project showcases different approaches to model building, data augmentation, and fine-tuning, providing a robust foundation for binary image classification tasks.

---

## Project Structure

- **Basic CNN Model:** A simple CNN model with three convolutional layers.
- **Improved CNN Model:** Enhanced CNN with batch normalization and dropout layers.
- **Advanced CNN Model:** A deeper CNN with four convolutional layers for better performance.
- **Transfer Learning with VGG16:** Fine-tuning a pre-trained VGG16 model for binary classification.
- **Data Augmentation:** Implementation of data augmentation to improve generalization.
- **Plots:** Visualizations of training and validation loss for performance monitoring.

---

## Requirements

To run this project, you need the following:

- Python 3.7+
- TensorFlow 2.0+
- Keras
- Matplotlib
- NumPy
- Pandas


---

## Implementation

### 1. Basic CNN Model

A basic CNN model with three convolutional layers followed by max pooling, fully connected layers, and a sigmoid activation function for binary classification.

#### Code:
```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

#### Training and Validation Loss Plot:
![Basic CNN Model Plot](images/Training-and-Validation-Loss_plot(Basicmodel).png)  

#### Training and Validation Accuracy Plot:
![Basic CNN Model Plot](images/Training-and-Validation-Accuracy_plot(Basicmodel).png)

---

### 2. Improved CNN Model

Added batch normalization and dropout to the CNN to improve training stability and reduce overfitting.

#### Code:
```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

#### Training and Validation Loss Plot:
![Improved CNN Model Plot](images/2nd-Training-and-Validation-Loss-plot.png)

---

### 3. Advanced CNN Model

A deeper CNN architecture with four convolutional layers and a larger fully connected layer for improved performance.

#### Code:
```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Conv2D(256, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

#### Training and Validation Loss Plot:
![Advanced CNN Model Plot](images/3rd-Training-and-Validation-Loss-plot.png)  

#### Training and Validation Accuracy Plot:
![Advanced CNN Model Plot](images/3rd-Training-and-Validation-Accuracy-plot.png)

---

### 4. Transfer Learning with VGG16

Using the pre-trained VGG16 model as the feature extractor and adding custom fully connected layers for binary classification.
#### Data Augmentation

Data augmentation techniques applied to training images for improved model generalization.

#### Code:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory='/content/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    directory='/content/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
```

#### Code:
```python


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Build the model
model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=320,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=90)
```
So we achive :  
Validation Loss: 0.16431619226932526  
Validation Accuracy: 0.9359999895095825

#### Training and Validation Loss Plot:
![Transfer Learning VGG16 Plot](images/VGG16-Training-and-Validation-Loss-plot.png)  

#### Training and Validation Accuracy Plot:
![Transfer Learning VGG16 Plot](images/VGG16-Training-and-Validation-Accuracy-plot.png)

---



## Results

The models achieve over 90% validation accuracy with proper tuning and augmentation. Fine-tuning the VGG16 model provides the best performance.

---

## Future Work

- Experiment with other architectures like ResNet and EfficientNet.
- Use techniques like Grad-CAM for better model explainability.
- Extend the project to multi-class classification.

---

## License
This project is open-source and licensed under the MIT License.

