# 1. Set Up the Environment
   Ensure you have the necessary software installed:

   Keras: Deep learning framework.
   Jupyter Notebook: For interactive development and model training.
   Visual Studio Code: Code editor for development.
   Flask: Web framework to integrate the model with the web-based system.
   TensorFlow: Machine learning framework compatible with Keras.
   Bootstrap: For the web interface design.

# 2. Install Necessary Packages
   You need to install the required Python packages. This can be done using pip. Open your terminal or command prompt and run the following commands:

       pip install tensorflow keras flask jupyter
       pip install bootstrap4

# 3. Prepare the Dataset
   Download the dataset from Kaggle and preprocess it. The dataset should include labeled facial expression images for training the model. You will use these images to train a Convolutional Neural Network (CNN)

# 4. Develop the Model
   Create and train the CNN model using Keras and TensorFlow. Here's my version of what the code look like:

# Import module
    import numpy as np 
    import pandas as pd 
    import tensorflow as tf 
    from tensorflow.keras.models import Sequential 
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
    from tensorflow.keras.preprocessing.image import ImageDataGenerator 

# Define the paths to train and test data directories
    train_data_dir = 'dataset/train'
    test_data_dir = 'dataset/validation'

# Define image dimensions and batch size
    img_width, img_height = 48, 48
    batch_size = 32

# Prepare the data using ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))  # Increased complexity
    model.add(Dropout(0.4))  # Adjusted dropout rate

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(5, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()  # Print the model summary

    epochs = 100  # adjust the number of epochs based on the training performance

    model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size)
    
# Save model
    model.save('emotion_model')  # Save the model in the native Keras format

# 5. Run the System
  # Start the Flask Application: Run your Flask application to start the web server
    python app.py
