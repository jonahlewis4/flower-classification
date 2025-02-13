import keras


DATASET_PATH = "flowers"
HEIGHT = 244
WIDTH = 244

####################################################################
# Load data from directory
####################################################################

#load_data_from_dir
def load_data_from_dir(directory: str) :
    #get training data from flowers directory
    training_dataset = keras.api.utils.image_dataset_from_directory(
        directory,  #path of where the data is. Each classification will be its own subdirectory
        validation_split=0.2,  #specifies that 20% of the data is used for validation. Remaining 80% is for training
        subset="training",
        #indicate that this dataset will be the training dataset (meaning get the 80% we want for training)
        seed=42,  #random seed. Using the same seed ensures the dataset is split the same each time.
        image_size=(HEIGHT, WIDTH),  #Resize all images to 224x224 pixels. They must be the same size to compare
        batch_size=32,  #the number of images per processing batch (think of this like a buffer).
    )

    #get validation data from flowers directory
    validation_dataset = keras.api.utils.image_dataset_from_directory(
        directory,  #path of where the data is. Each classification will be its own subdirectory
        validation_split=0.2,  #specifies that 20% of the data is used for validation. Remaining 80% is for training
        subset="validation",
        #indicate that this dataset will be the validation dataset (meaning get the 20% we want for validation)
        seed=42,  #random seed. Using the same seed ensures the dataset is split the same each time.
        image_size=(HEIGHT, WIDTH),  #Resize all images to 224x224 pixels. They must be the same size to compare
        batch_size=32,  #the number of images per processing batch (think of this like a buffer).
    )

    #get class names (daisy, dandelion, rose, sunflower, tulip)
    return training_dataset, validation_dataset, training_dataset.class_names