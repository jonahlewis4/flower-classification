from tensorflow.python.keras.utils.version_utils import training

from shared import HEIGHT, WIDTH, load_data_from_dir, DATASET_PATH



from keras.api.preprocessing import image
import numpy as np


#classify_image takes in a model, the path to an image, and the class_names, and returns the predicted class from class_names along with the confidence.
def classify_image(model, img_path, class_names):
    img = image.load_img(img_path, target_size=(HEIGHT, WIDTH)) #load the specified image and resize it to target_size
    img_array = image.img_to_array(img) / 255.0 #convert the image to NumPy array
    img_array = np.expand_dims(img_array, axis=0) #convert the numpy array into a batch of images of size 1. The model actually takes in batches of images

    predictions = model.predict(img_array) #run the model on the batch of 1.
    predicted_class = class_names[np.argmax(predictions)] #get class name associated with the largest confidence prediction
    confidence = np.max(predictions) #get confidence of the largest confidence prediction.

    return predicted_class, confidence

####################################################################
# Load the model
####################################################################
from keras.api.models import load_model
model = load_model("./flower_classifier.h5")
_, _, class_names = load_data_from_dir(DATASET_PATH) #get class names of the data set


import os
import random
def testRandomImage(model, class_names, dataset_path):
    # Pick a random class directory
    random_class_dir = random.choice(os.listdir(dataset_path))

    # Get the full path of the directory
    class_dir_path = os.path.join(dataset_path, random_class_dir)

    # Get a list of all image files in that directory
    image_files = [f for f in os.listdir(class_dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Pick a random image file from that directory
    random_image_file = random.choice(image_files)

    # Get the full path of the random image
    img_path = os.path.join(class_dir_path, random_image_file)

    # Classify the random image using the model
    predicted_class, confidence = classify_image(model, img_path, class_names)

    # Compare the predicted class with the actual class (directory name)
    correct = predicted_class == random_class_dir

    # Print the results
    if correct:
        print(f"Predicted: {predicted_class} Actual: {random_class_dir} Correct: \033[92m{correct}\033[0m")
    else:
        print(f"Predicted: {predicted_class} Actual: {random_class_dir} Correct: \033[91m{correct}\033[0m")

if __name__ == "__main__":
    for _ in range(0,1000) :
        testRandomImage(model, class_names, DATASET_PATH)

