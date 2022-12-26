### We create a bunch of helpful functions throughout the course.
### Storing them here so they're easily accessible.
import random
import os
import tensorflow as tf
import urllib
import datetime
import pandas as pd

# def create_tensorboard_callback(dir_name, experiment_name):
# def load_and_prep_image(filename, img_shape=224, scale=False):
# def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
# def def pred_and_plot(model, filename, class_names):
# def def create_tensorboard_callback(dir_name, experiment_name):
# def def plot_loss_curves(history):
# def def compare_historys(original_history, new_history, initial_epochs=5):
# 273 def unzip_data(filename):  
# 291 def walk_through_dir(dir_path):
# def def calculate_results(y_true, y_pred):
# def def create_model(model_url, num_classes=10, image_shape=(224,224)):
# def def create_list_name_classes(dir_data_train):
# def def view_random_image(target_dir, target_class):
# 424 def create_checkpoint_callback():
# 435 def data_augmentation()
# 466 def download_file()

def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instand to store log files.
    
    Stores log file with the filepath:
        "dir_name/experiment_name/current_datetime/"
    
    args:
        dir_name: target directory to store TensorBoard log 
        files experiment_name: name of experiment directory (e.g. efficientnet_B0)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224, scale=False):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  img = np.array(img,np.int32)  # la ligne qui m'a fait perdre bcp de temps
#   if scale:
#     # Rescale the image (get all values between 0 and 1) 
#     return img/255.
#   else:
  return img

# Note: The following confusion matrix code is a remix of Scikit-Learn's 
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")
  
# Make a function to predict on images and plot them (works with multi-class)

def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  if len(pred[0]) > 1: # check for multi-class
    pred_class = class_names[pred.argmax()] # if more than one output, take the max
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False);
  
import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

# Plot the validation and training data separately
import matplotlib.pyplot as plt

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
  
# Create function to unzip a zipfile into current working directory 
# (since we're going to be downloading and unzipping a few files)
import zipfile

def unzip_data(filename):
  """
  Unzips filename into the current working directory and delete it after unzip.

  Args:
    filename (str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, "r")
  zip_ref.extractall()
  zip_ref.close()
    
  if os.path.isfile(filename): # Remove the zip file after unzip
    os.remove(filename)

# Walk through an image classification directory and find out how many files (images)
# are in each subdirectory.
import os

def walk_through_dir(dir_path):
  """
  Walks through dir_path returning its contents.

  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
    
# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

# Import dependencies 
import tensorflow_hub as hub
from tensorflow.keras import layers

# Let's make a create_model() function to create a model from a URL
def create_model(model_url, num_classes=10, image_shape=(224,224)):
    """"
    Takes a TensorFlow Hub URL and creates a keras Sequential model with it

    args:
        model_url (str): A TensorFlow Hub feature extraction URL
        num_classes (int): Number of output neurons in the output layer,
        should be equal to number of target classes, default 10.
    Returns:
        An uncompiled Keras Sequential model with model_url as feature
        extractor layer and Dense output layer with num_classes output neurons.
    """
    # Download the pretrainded model and save it as a Keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,
                                             input_shape=image_shape + (3,))  # freeze the already learned patterns
    # Create our own model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(num_classes, activation="softmax", name="output_layer")
    ])

    return model


import pathlib
import numpy as np

# Create names classes as list  
def create_list_name_classes(dir_data_train):
    """
    args : directory of train_data, to extract names of folders in this directory
    """
    data_dir = pathlib.Path(dir_data_train)
    class_names = np.array(sorted([item.name for item in data_dir.glob("*")])) # Created a list of class_names from the subdirectories
    # class_names = class_names[1:] # Remove DS Store
    return class_names

def view_random_image(target_dir, target_class):
  # Set the target directory (we'll view images from here)
  target_folder = target_dir + target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)
  print(random_image)

  # Reading the image and plot in using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image 
  return img

import random
# load pred and plot random image
def load_pred_and_plot_random_image(model, target_dir, target_class, class_names):
  """
  Imports an random image located at target_dir/target_class, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  
  args : target_dir (train_dir or test dir)
  target_class (name of class dir, ex: pizza,ramen,...)
  """
  # Set the target directory (we'll view images from here)
  target_folder = target_dir + target_class

  # Get a random image path
  random_image_file_name = random.sample(os.listdir(target_folder), 1)
  print(random_image_file_name)
  # Import the target image and preprocess it
  img = load_and_prep_image(target_folder + "/" + random_image_file_name[0])

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  if len(pred[0]) > 1:
      pred_class = class_names[pred.argmax()] # if more than one output, take the max
  else:
      pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round
#   print(pred)
  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class},")
  plt.axis(False);

# Create a checkpoint
def create_checkpoint_callback(checkpoint="callbackcpoint"):
    checkpoint_filepath = checkpoint+'/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        save_best_only=True)
    return model_checkpoint_callback

# Create data augmentation layer
def data_augmentation(randomFlip="horizontal",randomRotation=0.2,randomHeight=0.2,randomWidth=0.2,
                      randomZoom=0.2,rescale=False):
    """
    Some operation to augmente our data
    NB: this function contains this list of operation:
        RandomFlip, randomRotation, randomHeight, randomWidth, randomZoom
    Default : 
    Rescale = False # make it True if you want rescale our data to 1/255. 
    RandomFlip = "horizontal",
    RandomRotation = 0.2
    RandomHeight = 0.2
    RandomZoom = 0.2
    """
    Rescaling = tf.keras.layers.experimental.preprocessing.RandomZoom(0.0)
    if rescale:
            Rescaling = tf.keras.layers.experimental.preprocessing.Rescaling(1/255.)
            
    data_augmentation_layer = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip(randomFlip),
        tf.keras.layers.experimental.preprocessing.RandomRotation(randomRotation), 
        tf.keras.layers.experimental.preprocessing.RandomHeight(randomHeight),
        tf.keras.layers.experimental.preprocessing.RandomWidth(randomWidth),
        Rescaling,
        tf.keras.layers.experimental.preprocessing.RandomZoom(randomZoom),
         
        # rescale inputs of images to between 0 & 1
    ], name="data_augmentation" )
    return data_augmentation_layer

# Download file 
def download_file(url, file_name, make_name_directory="no"):
    # Create a directory for our datasets
    """
    Download file
    If we want to choose directory we should change make_name_directory param to
    name of our directory we want create else that will download the file in current directory
    args : url : url of or file we want download
    file_name : file name
    """
    module_path = os.getcwd()
    if make_name_directory=="no":
        urllib.request.urlretrieve(url,file_name)
    else:
        path_datasets = os.path.join(module_path, make_name_directory)
        if os.path.isdir(path_datasets) == False:
            os.mkdir(path_datasets)
        os.chdir(path_datasets)
        urllib.request.urlretrieve(url,file_name)
        if os.path.isfile(file_name):
            print(f"{file_name} downloaded, current path {os.getcwd()}")
    os.chdir(module_path)
    
    # Create a function to load and prepare images
def load_and_prep_image_v2(filename, img_shape=224, scale=False):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_image(img,channels=3)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  img = np.array(img,np.int32)  # la ligne qui m'a fait perdre bcp de temps
  if scale:
     # Rescale the image (get all values between 0 and 1) 
    return img/255. #don't rescale the images for EfficientNet models in TensorFlow
  else:
    return img


def load_pred_and_plot_random_image_v2(model, target_dir, class_names):
    """
    Imports an random image located at target_dir/target_class, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.

    args : target_dir (train_dir or test dir)
    target_class (name of class dir, ex: pizza,ramen,...)
    """
    plt.figure(figsize=(17,10))
    for i in range(3):
        # Set the target directory (we'll view images from here)
        target_class = random.choice(class_names)
        target_folder = target_dir + target_class

        # Get a random image path
        random_image_file_name = random.sample(os.listdir(target_folder), 1)
        
        # Import the target image and preprocess it
        img = load_and_prep_image_v2(target_folder + "/" + random_image_file_name[0])

        # Make a prediction
        pred_prob = model.predict(tf.expand_dims(img, axis=0)) # get prediction proba array 
        if len(pred_prob[0]) > 1:
            pred_class = class_names[pred_prob.argmax()] # if more than one output, take the max
        else:
            pred_class = class_names[int(tf.round(pred_prob)[0][0])] # if only one output, round

#         pred_class = class_names[pred_prob.argmax()] # get and classe name with highest prediction proba 
        # Get the predicted class
        plt.subplot(1, 3, i+1)
        plt.imshow(img)
        
        if target_class == pred_class: # if predected class matches truth class make text green
            title_color = "g"
        else:
            title_color = "r"
        plt.title(f"actual:{target_class}, pred:{pred_class}, prob: {pred_prob.max():.2f}", c=title_color)
        plt.axis(False)
        
# Make a function for preprocessing images
def preprocess_img(image, label, img_shape=224):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes
    image to [img_shape, img_shape, colour_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape]) # reshape target image
    #image = image/255. # scale image values (not required with EfficientNetBX models from tf.keras.applications)
    
    label = tf.one_hot(label, num_classes)
    print("log : def preprocess_img(image, label, img_shape=224)")
    return tf.cast(image, tf.float32), label # return (float32_image, label) tuple

