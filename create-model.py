import tensorflow as tf
import keras
from tensorflow.python.data import AUTOTUNE

#constant definitions
from shared import DATASET_PATH, HEIGHT, WIDTH, load_data_from_dir

####################################################################
# Load data from directory
####################################################################

#get training data from flowers directory
training_dataset, validation_dataset, class_names = load_data_from_dir(DATASET_PATH)
print("Classes:", class_names)

####################################################################
# Normalize and Augment Data
####################################################################

AUTOTUNE = tf.data.AUTOTUNE  #AUTOTUNE is a helper object that automatically determines the optimal number of threads to use while normalizing the data


#normalize the data

def normalizer(image, the_class): #normalizer is used to map each dataset element to a normalized dataset element
    #normalize each image by dividing by 255 and storing as a tuple of normalized image and the target class.
    return image / 255.0, the_class


training_dataset.map(
    #normalize each image by dividing by 255 and storing as a tuple of normalized image and the target class.
    normalizer,
    num_parallel_calls=AUTOTUNE
)
validation_dataset.map(
    # normalize each image by dividing by 255 and storing as a tuple of normalized image and the target class.
    normalizer,
    num_parallel_calls=AUTOTUNE
)

#augment the data


#Sequentail will apply each augmentation in the passed array to the dataset in sequential order.
#the goal of the transformations is to increase the diversity of the dataset and help improve model generalization
#we're basically making the data slightly 'worse' to prepare for that type of data in the real world
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),  #flips the image horizontally sometimes
    keras.layers.RandomRotation(.1), #rotates the image between -36 and 36 degrees
    keras.layers.RandomZoom(0.1), #zoom in or out between 90-100%.
])


####################################################################
# Build CNN Model
####################################################################

from keras import models, layers
#models.Sequential will run each model in the array in sequential order.

#there are already existing models out there that might be worth looking into.

#define the model
model = models.Sequential([
    layers.InputLayer(input_shape=(HEIGHT, WIDTH, 3)), #Input layer specifies the shape of the data. It doesn't actually perform any computation.

    data_augmentation, #perform data augmentation specified previously

    #convolutional layers (actual machine learning of features. With each call the model learns more.
    # The first layer might pick up on edges.
    # The next layer might figure out the shapes that those edges make.
    # The third layer might identify what sort of flower is made up of those shapes.

    #generally the deeper we get in layers, the more abstract and complicated the pattern gets.
    #usually the number of filters will increase accordingly.

    #activation="relu" will allow for  non-linearity, which allows for patterns that fit the data better.

    #more details on convolutional layers can be found at https://www.geeksforgeeks.org/what-are-convolution-layers/

    #the number of layers is something that involved fine-tuning.
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(), #MaxPooling2D reduces the dimensions of the features/patterns. Since images are 2d, we don't want to work with higher dimensional data. That would be more expensive

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(), #flatten the 2d layers into a 1d vector
    layers.Dense(128, activation = "relu"), #A dense layer is another ML magic thing. Generally they make final decisions, like classifying the image
    layers.Dropout(0.5), #layers.Dropout does some magic to try to prevent overfitting
    layers.Dense(len(class_names), activation="softmax")
])
#compile the model
model.compile(
    optimizer="adam", #an optimizer controls how weights in model are updated during training
    loss="sparse_categorical_crossentropy", #loss measures the mdoels accuracy. sparse_categorical_crossentropy is typically used when identifying classes
    metrics=["accuracy"] #metrics indicates which metrics to actually measure while running the model
)

model.summary() #summary outputs a summary of the ML model.


####################################################################
# Train the model
####################################################################

#train the model for 20 epochs
history = model.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = 20
)

####################################################################
# Save the model
####################################################################
model.save("flower_classifier.h5")

#model can be loaded by filename like this:
#from keras.api.models import load_model
#model = load_model(<name_of_model>)

