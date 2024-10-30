import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,roc_curve,auc
import shutil
import random
import glob
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array,ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Flatten, Dense,Conv2D, MaxPooling2D


def preprocess_image(image_path):
    
    image_path = image_path
    img = load_img(image_path, target_size=(224, 224))  # Match target size to VGG16
    img_array = img_to_array(img)  # Convert to array
    img_array = preprocess_input(img_array)  # Preprocess for VGG16
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def imagedata(data_path, batch_size=16, target_size=(224,224),shuffle=True):
    
    return (ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
            .flow_from_directory(directory=data_path,target_size=target_size,classes=['cat','dog'],
                                 batch_size=batch_size,shuffle=shuffle))

# plot images i nthe form of a grid with 3 row and 4 columns where images are placed
def plotimages (images_batches):
    
    imgs, labels = next(images_batches)
    fig, axes = plt.subplots(4,4,figsize=(20,20))
    axes=axes.flatten()
    for img,ax in zip(imgs, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/Samples of Cat and Dog Images.png')
    plt.show()
    return labels
    
def create_model(imgs):
    
    return (Sequential([Input(shape=(imgs.shape[1], imgs.shape[2],imgs.shape[3])),
                        Conv2D(32, kernel_size=(3, 3), activation='relu',padding='same'),
                        MaxPooling2D(pool_size=(2, 2),strides=2),
                        Conv2D(64, kernel_size=(3, 3), activation='relu',padding='same'),
                        MaxPooling2D(pool_size=(2, 2),strides=2),
                        Flatten(),
                        Dense(2, activation='softmax')
                       ]))


# The confusion matrix function
def confus_matrix(model, test_batches, title):
    
    # Obtain the predictions against actual
    y_test = test_batches.classes
    y_pred = np.argmax(model.predict(x=test_batches,verbose=0),axis=-1)
    plt.figure(figsize=(7,5))

    # Obtain the Confusion matrix
    cm = confusion_matrix(y_test,y_pred,normalize = 'true')
    sns.heatmap(cm,annot=True,fmt=".0%")
    plt.title(title)
    plt.ylabel('Actual')
    plt.xticks(np.arange(0.5, 2,1),['Cat','Dog'],rotation=90)
    plt.yticks(np.arange(0.5,2,1), ['Cat','Dog'],rotation=0)
    plt.xlabel('Predicted')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/Confusion Matrix for {title}.png')
    plt.show()

# Save the model
def save_model(model, model_name):
    os.makedirs('./model', exist_ok=True)
    # store the model as model_name.pkl
    model.save(f'model/{model_name}.keras')



# Function to plot ROC
def plot_roc_curve(model, test_batches):
    
    # Generate predictions (probabilities) for the test data
    y_true = test_batches.classes  # True labels
    y_pred_proba = model.predict(test_batches)[:, 1]  # Probability for class 1
    
    # Compute ROC curve and ROC area for the binary classification
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange",lw=2, label=f"ROC curve (area = {roc_auc:.2%})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2,linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="best")
    plt.grid()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/ROC Curve.png')
    plt.show()


# Save the individual dataset
def move_images(train_folder="", test_folder="", validate_folder="", source=""):
    
    # Validate that source is provided
    if not source:
        raise ValueError("Source folder is required")
        
    # Ensure source directories exist
    if not os.path.isdir(source):
        raise ValueError(f"Source folder '{source}' does not exist")
        
    # Ensure target folder names are given
    if not all([train_folder, test_folder, validate_folder]):
        raise ValueError("Train, test, and validate folder names are required")
        
    # If the subfolders already exist, skip and do nothing  
    if os.path.isdir(f'./images/{train_folder}/Dog'):
        return
        
    # List of folders with sample sizes for train, test, and validation sets
    data_folders = {
        train_folder: 500,
        test_folder: 100,
        validate_folder: 100
    }

    # Create target folders
    os.makedirs("images", exist_ok=True)

    for folder, sample_size in data_folders.items():
        # Create subdirectories for each class
        for label in ['dog', 'cat']:
            os.makedirs(f'./images/{folder}/{label}', exist_ok=True)

            # Get list of images and move samples
            image_paths = glob.glob(f'./{source}/{label.capitalize()}/*.jpg')
            if len(image_paths) < sample_size:
                print(f"Warning: Not enough images for '{label}' in '{source}'. Required: {sample_size}, Found: {len(image_paths)}")
                sample_size = len(image_paths)  # Adjust to available images if fewer than sample size
                
            for img_path in random.sample(image_paths, sample_size):
                shutil.move(img_path, f'./images/{folder}/{label}')

def plot_cnn_history(history,model_name):
    
    # Ensure history is from a trained model
    if not hasattr(history, 'history'):
        print("Error: Invalid history object")
        return

    # Get the training history
    history_data = history.history  # This is a dictionary

    # Create plots/ directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Check if necessary keys are present in history
    required_keys = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    for key in required_keys:
        if key not in history_data:
            print(f"Warning: '{key}' not found in history data.")
            return

    # Plot the loss and accuracy
    plt.figure(figsize=(12, 8))
    
    # Plotting training loss and accuracy
    plt.plot(history_data['loss'], "r-", label="Train Loss")
    plt.plot(history_data['accuracy'], "g-", label="Train Accuracy")
    
    # Plotting validation loss and accuracy
    plt.plot(history_data['val_loss'], "r--", label="Validation Loss")
    plt.plot(history_data['val_accuracy'], "g--", label="Validation Accuracy")

    # Labeling and formatting
    plt.title(f"{model_name} Loss and Accuracy Over Epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.ylim(0, 1)  # Set y-axis limit for better readability
    plt.legend(loc='best')
    plt.grid(visible=True, linestyle="--", alpha=0.7)
    
    # Save plot
    plt.savefig(f'plots/{model_name} History.png')
    plt.show()