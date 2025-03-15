import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from keras.models import load_model,Sequential,Model
from keras.layers import Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow_addons as tfa
import shutil	
import pandas as pd
import numpy as np
import json
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.style.use('default')
print(plt.style.available)
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import PIL
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from tqdm.auto import tqdm
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optuna.integration import KerasPruningCallback
import wandb
from wandb.integration.keras import WandbCallback,WandbMetricsLogger
os.environ["WANDB_MODE"] = "offline"
from datetime import datetime
import math
#suppress Warnings
import warnings
warnings.filterwarnings("ignore")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import onnxmltools


def check_gpu_availability():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
check_gpu_availability() 

print('*'*50)
print('you are using the general_ML_utils module')
print('*'*50)
    
    
def one_hot_encode(labels, n_classes, classification_type='multiclass'):
    '''
    This function one-hot encodes the labels for multiclass or multilabel classification.
    It takes the labels, the number of classes, and the classification type as input and returns the one-hot encoded labels.
    For multiclass inputs, the labels should look like [0,1,2,3,4,5,6,7,8,9]  
    For multilabel inputs, the labels should look like [[1,3],[2],[0,1,2,3],[4,5,6]] 

    Parameters:
    labels: numpy array or list, labels for the dataset
    n_classes: int, number of classes in the dataset
    classification_type: str, either 'multiclass' or 'multilabel'

    Returns:
    one_hot_labels: numpy array, one-hot encoded labels for the dataset
    '''
    if classification_type == 'multiclass':
        one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=n_classes)
    elif classification_type == 'multilabel':
        one_hot_labels = np.zeros((len(labels), n_classes))
        for i, label_set in enumerate(labels):
            one_hot_labels[i, label_set] = 1
    else:
        raise ValueError("classification_type must be either 'multiclass' or 'multilabel'")

    return one_hot_labels

def build_img_dataset_pipeline(img_path, label_path, n_classes, img_size=None, batch_size = 16, augment_strength=0.1, val_size=0.2, test_size=0.1, shuffle=True, img_type='png',seed=42, verbose=1, return_filenames=False):
    '''
    This function builds the training pipeline for image data. It reads the images from the image_path and labels from the label_path.
    It then performs the following operations:
    1. Reads the images and labels
    2. Resizes the images to img_size
    3. Augments the images
    4. Splits the data into training, validation and test sets
    5. Creates a tf.data.Dataset object for each set
    6. Batches the data
    7. Shuffles the data
    8. Returns the training, validation and test datasets
    
    Parameters:
    img_path: str, path to the folder that contains the images; images should be in .png or .jpeg format
    label_path: str, path to the file that contains the labels; labels should be in .csv format where the first column contains the image names and the other columns contain the labels for each class;
                the labels should be in the format: one-hot encoded so (1,0,0) for class 1, (0,1,0) for class 2, (0,0,1) for class 3, etc. for multiclass classification or (1,1,0) for class 1 and 2, (0,1,1) for class 2 and 3, etc. for multilabel classification
    n_classes: int, number of classes in the dataset
    img_size: tuple (optional, default: None), size to which the images should be resized; should be in the format (height, width, channels); if None using original size 
    batch_size: int (optional, default: 16), batch size for the training, validation and test datasets
    augment_strength: float/list (optional, default: 0.1), strength of the augmentation; should be between >0 and 1; 
                    can be one float for all augmentations or a list with a value for each augmentation,
                    if list must contain the same number of values as augmentations (6);
                    the augmentations are: 
                    - random flip, (possible parameters are 0,1,2,3 for no flip, horizontal and vertical flip, horizontal flip, vertical flip)
                    - random translation horizontal (parameter between 0-1; is the percentage of the image size ), 
                    - random translation vertical (parameter between 0-1; is the percentage of the image size), 
                    - random contrast (parameter between 0-1; is the percentage of the maximum addable or subtractable contrast),
                    - random brightness (parameter between 0-1; is the percentage of the maximum addable or subtractable brightness), 
                    - gaussian noise (parameter between 0-1; is the strength of the noise)
    val_size: float (optional, default: 0.2), percentage size of the validation set; should be between >0 and 1
    test_size: float (optional, default: 0.1), percentage size of the test set; should be between 0 and 1, if no test set is needed, set it to 0
    shuffle: bool (optional, default: True), whether to shuffle the data
    img_type: str (optional, default: 'png'), type of the images; should be 'png' or 'jpeg'
    seed: int (optional, default: 42), seed for shuffling the data, same seed will give same shuffling order
    verbose: int (optional, default:1), whether to show helpful messages
    return_filenames: bool (optional, default: False), whether to return the filenames of the images ussed in training, validation and test datasets
    
    Returns:
    train_ds: tf.data.Dataset, training dataset
    val_ds: tf.data.Dataset, validation dataset
    test_ds: tf.data.Dataset, test dataset
    if return_filenames==True returns additionally:
    train_files: list, list of filenames used in the training dataset
    val_files: list, list of filenames used in the validation dataset
    test_files: list, list of filenames used in the test dataset

    (tf.data.Dataset objects are used to build efficient input pipelines for TensorFlow models,
    they consist of a sequence of elements with each element consisting of image and label. The elements can be iterated over using the 'for' loop.
    tf and keras Models can use these datasets as input using the 'fit' method directly)
    '''
    assert os.path.exists(img_path), 'image_path does not exist'
    assert os.path.exists(label_path), 'label_path does not exist'
    assert len(img_size) == 3, 'img_size should have 3 dimensions, last dimension contains image channels'
    assert isinstance(augment_strength, (float,list)), 'augment_strength should be a float or a list of floats'
    if isinstance(augment_strength, list):
        assert len(augment_strength) == 6, 'augment_strength should contain 6 values; one for each augmentation'
        assert augment_strength[0] in [0,1,2,3], 'augment_strength[0] should be either 0,1,2,3 for no flip, horizontal and vertical flip, horizontal flip, vertical flip'
    else:
        augment_strength = [augment_strength]*5
        #add 1 as first element for random flip
        augment_strength.insert(0, 1)
    assert img_type in ['png','jpeg'], 'img_type should be either "png" or "jpeg"'
    print('building training pipeline ...')
    
    #define a function to read the images from image filepaths
    def parse_images(image_path, label):
        image = tf.io.read_file(image_path)
        if img_type == 'png':
            image = tf.image.decode_png(image, channels=img_size[2])
        elif img_type == 'jpeg':
            image = tf.image.decode_jpeg(image, channels=img_size[2])
        image = tf.image.resize(image, (img_size[0],img_size[1]),method='area')
        return image, label
    
    labels = pd.read_csv(label_path)
    img_files = labels.iloc[:,0].values 
    img_files = img_path + '\\' + img_files #make whole path for images
    labels = labels.drop(labels.columns[0], axis=1)
    class_names = labels.columns
    labels = labels.to_numpy()
    assert len(img_files) == len(labels), 'number of images and labels do not match'
    assert n_classes == labels.shape[1], 'number of classes does not match the number of classes in the labels'
    image_of_each_class ={}
    k=0
    if verbose:
        while len(image_of_each_class) < n_classes:
            for i in range(len(labels)):
                label = np.argmax(labels[i])
                if label not in image_of_each_class.keys():
                    image_of_each_class[label] = img_files[i]
                k+=1
                if k >= len(labels):
                    break
            if k >= len(labels):
                    break
            
    dataset = tf.data.Dataset.from_tensor_slices((img_files,labels))#construct tf.data.Dataset object
    if shuffle:
        dataset = dataset.shuffle(len(img_files), reshuffle_each_iteration=False,seed=seed)
    
    #if one size is 1.0 set the others to 0.0
    if val_size == 1.0:
        test_size = 0.0
    if test_size == 1.0:
        val_size = 0.0
    #split dataset into train and validation and test sets
    test_nr = int(test_size * len(img_files))
    val_nr = int(val_size * len(img_files))
    train_nr = int((1-val_size-test_size) * len(img_files))
    print('nr of training samples:',train_nr)
    print('nr of validation samples:',val_nr)
    print('nr of test samples:',test_nr)
    if train_nr == 0:
        train_ds = None
    else:
        train_ds = dataset.take(train_nr)
    remaining_ds = dataset.skip(train_nr) # Remaining after substracting train_ds
    if test_size:
        val_ds = remaining_ds.take(val_nr) if val_size > 0 else None # Take val_nr items to get val_ds
        test_ds = remaining_ds.skip(val_nr) # Skip val_nr items to get test_ds
    else:
        val_ds = remaining_ds if val_size > 0 else None # If no test set, remaining is all validation
        test_ds = None
    
    
    if verbose:
        if train_ds:
            train_files, train_labels = zip(*train_ds.as_numpy_iterator())
            cls_dict_train = get_dataset_composition(train_labels,class_names,'train')
        if val_ds:
            val_files, val_labels = zip(*val_ds.as_numpy_iterator())
            cls_dict_val = get_dataset_composition(val_labels,class_names,'validation')
        if test_ds:
            test_files, test_labels = zip(*test_ds.as_numpy_iterator())
            cls_dict_test = get_dataset_composition(test_labels,class_names,'test')
    
    if return_filenames:
        print('extracting filenames ...')
        train_files, val_files, test_files = [],[],[]
        if train_ds:
            for img, label in tqdm(train_ds.as_numpy_iterator()):
                #get only the filename without the path
                img = img.decode('utf-8').split('\\')[-1]
                train_files.append(img)
        if val_ds:
            for img, label in val_ds.as_numpy_iterator():
                img = img.decode('utf-8').split('\\')[-1]
                val_files.append(img)
        else:
            val_files = None
        if test_ds:
            for img, label in test_ds.as_numpy_iterator():
                img = img.decode('utf-8').split('\\')[-1]
                test_files.append(img)
        else:
            test_files = None
            
    #define a object containing augmentation layers; is used for augmenting the images in the training dataset
    #siehe https://keras.io/2.16/api/layers/preprocessing_layers/image_augmentation/ für Anpassungen und weitere Optionen
    def custom_random_flip(x, flip_param):
        if flip_param == 0:  # No flip
            return x
        elif flip_param == 1:  # Horizontal and vertical
            return tf.image.random_flip_left_right(tf.image.random_flip_up_down(x))
        elif flip_param == 2:  # Only horizontal
            return tf.image.random_flip_left_right(x)
        elif flip_param == 3:  # Only vertical
            return tf.image.random_flip_up_down(x)
        else:
            raise ValueError("Invalid flip parameter. Must be 0, 1, 2, or 3.")
        
    trainAug = Sequential([ tf.keras.layers.Lambda(lambda x: custom_random_flip(x, augment_strength[0])),
                            tf.keras.layers.RandomTranslation(height_factor=augment_strength[1], width_factor=augment_strength[2], fill_mode='reflect',interpolation="bilinear"),#translation by % of the image size, not applicable for all datasets, uncomment if needed	
                            tf.keras.layers.RandomContrast(augment_strength[3]),
                            tf.keras.layers.RandomBrightness(augment_strength[4]),
                            tf.keras.layers.GaussianNoise(augment_strength[5]),
                            tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0, 255)),#clip values to [0,1]
                            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.uint8)),#cast to int8
                            ])
    if train_ds:
        train_ds = train_ds.map(parse_images, num_parallel_calls=tf.data.AUTOTUNE)# here we read the images from the file; happens during training for reduced memory usage and faster training
        train_ds = train_ds.map(lambda image, label: (trainAug(image,training=True), label), num_parallel_calls=tf.data.AUTOTUNE)# here we augment the images
        train_ds = train_ds.batch(batch_size)#batch the data
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)#prefetch the data for faster training
    if val_ds:
        val_ds = val_ds.map(parse_images, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    if test_size:
        test_ds = test_ds.map(parse_images, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size)
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
    if verbose:
        #plot the images
        plt.figure(figsize=(20,10))
        print('number of classes:',n_classes)
        for i in range(n_classes):
            plt.subplot(2,math.ceil(n_classes/2), i+1)
            if i not in image_of_each_class.keys():
                #make white image
                img = np.ones((img_size[0],img_size[1],img_size[2]))
            else:
                img_file = image_of_each_class[i]
                img = PIL.Image.open(img_file)
                #resize to img_size
                img = img.resize((img_size[1],img_size[0]))
            cmap= 'gray' if img_size[2] == 1 else None
            plt.imshow(img,cmap=cmap)
            plt.axis('off')
            plt.title(f'{class_names[i]}')
        plt.suptitle('Images from each class in the dataset')
        plt.show()
        
        if train_ds:
            #plot the augmented images
            plt.figure(figsize=(20,10))
            for i in range(n_classes):
                plt.subplot(2,math.ceil(n_classes/2), i+1)
                if i not in image_of_each_class.keys():
                    #make white image
                    img = np.ones((img_size[0],img_size[1],img_size[2]))
                else:
                    img_file = image_of_each_class[i]
                    img = PIL.Image.open(img_file)
                    img = img.resize((img_size[1],img_size[0]))
                    img = trainAug(np.array(img),training=True)
                cmap= 'gray' if img_size[2] == 1 else None
                plt.imshow(img,cmap=cmap)
                plt.axis('off')
                plt.title(f'{class_names[i]}')
            plt.suptitle(f'Augmented Images with augment strength = {augment_strength} from each class in the dataset')
            plt.show()
        
    if return_filenames:
        return train_ds, val_ds, test_ds, train_files, val_files, test_files   
    else:
        return train_ds, val_ds, test_ds


def save_augmented_dataset(img_path, label_path, save_path, augment_strength, augment_iterations, img_size, img_type, n_classes):
    '''
    This function saves the augmented dataset to the save_path. It reads the images from the ds_path and labels from the label_path.
    It then performs the following operations:
    1. Reads the images and labels
    2. Resizes the images to img_size
    3. Augments the images
    4. copy the original images to the save_path
    5. Saves the additional augmented images to the save_path
    6. Saves the labels to the save_path with the labels for the augmented images
    
    Parameters:
    img_path: str, path to the folder that contains the images; images should be in .png or .jpeg format
    label_path: str, path to the file that contains the labels; labels should be in .csv format where the first column contains the image names and the other columns contain the labels for each class
    save_path: str, path to the folder where the augmented dataset should be saved; the folder will be created if it does not exist
    augment_strength: float, strength of the augmentation; should be between 0 and 1
    augment_iterations: int, number of times the whole original dataset should be augmented; 3 means the dataset will be quadrupled;
                        each augmented image will be named as 'original_image_name_aug{i}.png' where i is the iteration number
    img_size: tuple, size to which the images should be resized; should be in the format (height, width, channels)
    class_names: list, list of class names
    '''
    #copy all images from the dataset to the save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path,'images')):
        os.makedirs(os.path.join(save_path,'images'))
       
    labels = pd.read_csv(label_path)
    img_names = labels.iloc[:,0].values
    columns = labels.columns
    
    train_ds,_,_ = build_img_dataset_pipeline(img_path, label_path, img_size=img_size, n_classes=n_classes, batch_size = 1, augment_strength=augment_strength, val_size=0, test_size=0, shuffle=False, img_type=img_type, verbose=0)
    print('saving augmented images ...')
    new_rows=[]
    for i in tqdm(range(augment_iterations)):
        j=0
        for image, label in tqdm(train_ds.as_numpy_iterator()):
            img =tf.keras.preprocessing.image.array_to_img(image[0,:,:,:])
            img_name = f'{img_names[j].split(".png")[0]}_aug{i}.png'
            image_path = os.path.join(save_path,'images', img_name)
            img.save(image_path)
            new_row = labels.iloc[j].copy()
            new_row[columns[0]] = img_name
            new_rows.append(new_row)
            j+=1
    labels = pd.concat([labels] + new_rows, axis=0, ignore_index=True)
    labels.to_csv(os.path.join(save_path,'labels.csv'),index=False)
    print('augmented dataset saved to:',save_path)
    
    
model_min_img_sizes = {'resnet':{ 's': 32, 'm':32, 'l':32}, 'inceptionv3':{ 's': 75, 'm':75, 'l':75}, 'mobilenetv2':{ 's': 32, 'm':32, 'l':32}, 'xception':{ 's': 71, 'm':71, 'l':71}, 'efficientnet':{ 's': 32, 'm':32, 'l':32}, 'nasnet':{ 's': 32,'m':None, 'l':331}, 'simplenet':{ 's': 8, 'm':32, 'l':64}}

def build_model(input_shape, n_classes, classification_type, model_type='resnet',model_size='s', weights=None, trainable_layers=3, classifier_neurons=[512,128], dropout=0.4, l2_reg=0.01,optimizer ='adam', lr=0.01, verbose=1):
    '''
    This function builds a model for image classification. It uses the Keras API.
    
    Parameters:
    input_shape: tuple, shape of the input images; should be in the format (height, width, channels)
    num_classes: int, number of classes
    classification_type: str, type of classification; should be 'multiclass' or 'multilabel; multiclass is used when only one class can be present in the image, 
                        multilabel is used when multiple classes can be present in the image'
    model_type: str (optional, default: 'resnet'), type of the model to use; should be one of the following: 'resnet', 'inceptionv3', 'mobilenetv2', 'xception','efficientnet', 'nasnet','simplenet'
    model_size: str (optional, default: 's'), size of the model to use; should be one of the following: 's', 'm', 'l'
    weights: str (optional, default: None), weights to use for the pretrained model; should be 'imagenet' or None
    trainable_layers: int (optional, default: 3), number of layers to make trainable in the pretrained model, only applies if weights='imagenet; should be between 0 and 5
    classifier_neurons: list (optional, default: [512,128]), list of integers, each integer defines the number of neurons in a layer of the classifier
    dropout: float (optional, default: 0.4), dropout rate for the model, should be between 0 and 1; used to prevent overfitting
    l2_reg: float (optional, default: 0.01), l2 regularization parameter for the model, should be between 0 and 1; used to prevent overfitting
    optimizer: str (optional, default: 'adam'), optimizer to use for the model; should be one of the following: 'adam', 'rmsprop', 'lamb'
    lr: float (optional, default: 0.01), learning rate for the optimizer
    verbose: int, whether to print the summary of the model
    
    Returns:
    model: keras.Model, the model for image classification
    
    '''
    assert model_type in ['resnet', 'inceptionv3', 'mobilenetv2', 'xception','efficientnet','nasnet','simplenet'], 'model_type should be one of the following: resnet, vgg, inceptionv3, mobilenetv2, xception, efficientnet'
    assert weights in ['imagenet', None], 'weights should be one of the following: imagenet, None'
    assert model_size in ['s', 'm', 'l'], 'model_size should be one of the following: s, m, l'
    assert len(classifier_neurons) > 0, 'classifier_neurons should contain at least one element'
    assert classification_type in ['multiclass','multilabel'], 'classification_type should be either "multiclass" or "multilabel"'
    if weights == 'imagenet':
        assert input_shape[-1] == 3, 'input_shape should have 3 channels for imagenet weights'
    
    inputs = Input(shape=input_shape, name='input')
    if model_type == 'resnet':
        assert input_shape[0] >=32 and input_shape[1] >= 32, 'image size should be at least 32x32 for resnet'
        preprocess = tf.keras.applications.resnet_v2.preprocess_input
        inputs = preprocess(inputs)
        match model_size:
            case 's':
                base_model = tf.keras.applications.ResNet50V2(include_top=False, weights=weights, input_tensor=inputs)
            case 'm':
                base_model = tf.keras.applications.ResNet101V2(include_top=False, weights=weights, input_tensor=inputs)
            case 'l':
                base_model = tf.keras.applications.ResNet152V2(include_top=False, weights=weights, input_tensor=inputs)
    elif model_type == 'efficientnet':
        assert input_shape[0] >=32 and input_shape[1] >= 32, 'image size should be at least 32x32 for efficientnet'
        preprocess = tf.keras.applications.efficientnet.preprocess_input
        inputs = preprocess(inputs)
        match model_size:
            case 's':
                base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=weights, input_tensor=inputs)
            case 'm':
                base_model = tf.keras.applications.EfficientNetB1(include_top=False, weights=weights, input_tensor=inputs)
            case 'l':
                base_model = tf.keras.applications.EfficientNetB2(include_top=False, weights=weights, input_tensor=inputs)
    elif model_type == 'inceptionv3':
        assert input_shape[0] >=75 and input_shape[1] >= 75, 'image size should be at least 75x75 for inceptionv3'
        preprocess = tf.keras.applications.inception_v3.preprocess_input
        inputs = preprocess(inputs)
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights=weights, input_tensor=inputs)
    elif model_type == 'xception':
        assert input_shape[0] >=71 and input_shape[1] >= 71, 'image size should be at least 71x71 for xception'
        preprocess = tf.keras.applications.xception.preprocess_input
        inputs = preprocess(inputs)
        base_model = tf.keras.applications.Xception(include_top=False, weights=weights, input_tensor=inputs)
    elif model_type == 'mobilenetv2':
        assert input_shape[0] >=32 and input_shape[1] >= 32, 'image size should be at least 32x32 for mobilenetv2'
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        inputs = preprocess(inputs)
        base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=weights, input_tensor=inputs)
    elif model_type == 'nasnet':
        assert input_shape[0] >=32 and input_shape[1] >= 32, 'image size should be at least 32x32 for nasnet'
        preprocess = tf.keras.applications.nasnet.preprocess_input
        inputs = preprocess(inputs)
        match model_size:
            case 's':
                base_model = tf.keras.applications.NASNetMobile(include_top=False, weights=weights, input_tensor=inputs)
            case 'm':
                raise ValueError('NASNet only available in size s and l')
            case 'l':
                assert input_shape[0] >=331 and input_shape[1] >= 331, 'image size should be at least 331x331 for nasnet'
                base_model = tf.keras.applications.NASNetLarge(include_top=False, weights=weights, input_tensor=inputs)
    elif model_type == 'simplenet':
        assert weights is None, 'simplenet does not support pretrained weights'
        preprocess = tf.keras.layers.Rescaling(1./255)
        input = preprocess(inputs)
        match model_size:
            case 's':
                #make a simple model with 3 convolutional layers
                x = tf.keras.layers.Conv2D(32, (3,3), activation='elu', padding='same', name='conv1')(input)
                x = tf.keras.layers.BatchNormalization(name='bn1')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool1')(x)
                x = tf.keras.layers.Dropout(0.25, name='dropout1')(x)
                
                x = tf.keras.layers.Conv2D(64, (3,3), activation='elu', padding='same', name='conv2')(x)
                x = tf.keras.layers.BatchNormalization(name='bn2')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool2')(x)
                x = tf.keras.layers.Dropout(0.25, name='dropout2')(x)
                
                x = tf.keras.layers.Conv2D(128, (3,3), activation='elu', padding='same', name='conv3')(x)
                x = tf.keras.layers.BatchNormalization(name='bn3')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool3')(x)
                
                base_model = Model(inputs=inputs, outputs=x)
                
            case 'm':
                assert input_shape[0] >=16 and input_shape[1] >= 16, 'image size should be at least 32x32 for this model'
                #make a simple model with 5 convolutional layers
                x = tf.keras.layers.Conv2D(32, (3,3), activation='elu', padding='same', name='conv1')(input)
                x = tf.keras.layers.BatchNormalization(name='bn1')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool1')(x)
                x = tf.keras.layers.Dropout(0.1, name='dropout1')(x)
                
                x = tf.keras.layers.Conv2D(64, (3,3), activation='elu', padding='same', name='conv2')(x)
                x = tf.keras.layers.BatchNormalization(name='bn2')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool2')(x)
                x = tf.keras.layers.Dropout(0.2, name='dropout2')(x)
                
                x = tf.keras.layers.Conv2D(128, (3,3), activation='elu', padding='same', name='conv3')(x)
                x = tf.keras.layers.BatchNormalization(name='bn3')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool3')(x)
                x = tf.keras.layers.Dropout(0.3, name='dropout3')(x)
                
                x = tf.keras.layers.Conv2D(256, (3,3), activation='elu', padding='same', name='conv4')(x)
                x = tf.keras.layers.BatchNormalization(name='bn4')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool4')(x)
                x = tf.keras.layers.Dropout(0.4, name='dropout4')(x)
                
                x = tf.keras.layers.Conv2D(512, (3,3), activation='elu', padding='same', name='conv5')(x)
                x = tf.keras.layers.BatchNormalization(name='bn5')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool5')(x)
                
                base_model = Model(inputs=inputs, outputs=x)
                
            case 'l':
                assert input_shape[0] >=64 and input_shape[1] >= 64, 'image size should be at least 64x64 for this model'
                x = tf.keras.layers.Conv2D(32, (3,3), activation='elu', padding='same', name='conv1')(input)
                x = tf.keras.layers.BatchNormalization(name='bn1')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool1')(x)
                x = tf.keras.layers.Dropout(0.1, name='dropout1')(x)
                
                # Layer 2
                x = tf.keras.layers.Conv2D(64, (3,3), activation='elu', padding='same', name='conv2')(x)
                x = tf.keras.layers.BatchNormalization(name='bn2')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool2')(x)
                x = tf.keras.layers.Dropout(0.1, name='dropout2')(x)
                
                # Layer 3
                x = tf.keras.layers.Conv2D(128, (3,3), activation='elu', padding='same', name='conv3')(x)
                x = tf.keras.layers.BatchNormalization(name='bn3')(x)
                x = tf.keras.layers.Dropout(0.2, name='dropout3')(x)
                
                # Layer 4
                x = tf.keras.layers.Conv2D(128, (3,3), activation='elu', padding='same', name='conv4')(x)
                x = tf.keras.layers.BatchNormalization(name='bn4')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool4')(x)
                x = tf.keras.layers.Dropout(0.2, name='dropout4')(x)
                
                # Layer 5
                x = tf.keras.layers.Conv2D(256, (3,3), activation='elu', padding='same', name='conv5')(x)
                x = tf.keras.layers.BatchNormalization(name='bn5')(x)
                x = tf.keras.layers.Dropout(0.3, name='dropout5')(x)
                
                # Layer 6
                x = tf.keras.layers.Conv2D(256, (3,3), activation='elu', padding='same', name='conv6')(x)
                x = tf.keras.layers.BatchNormalization(name='bn6')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool6')(x)
                x = tf.keras.layers.Dropout(0.3, name='dropout6')(x)
                
                # Layer 7
                x = tf.keras.layers.Conv2D(512, (3,3), activation='elu', padding='same', name='conv7')(x)
                x = tf.keras.layers.BatchNormalization(name='bn7')(x)
                x = tf.keras.layers.Dropout(0.4, name='dropout7')(x)
                
                # Layer 8
                x = tf.keras.layers.Conv2D(512, (3,3), activation='elu', padding='same', name='conv8')(x)
                x = tf.keras.layers.BatchNormalization(name='bn8')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool8')(x)
                x = tf.keras.layers.Dropout(0.4, name='dropout8')(x)
                
                # Layer 9
                x = tf.keras.layers.Conv2D(1024, (3,3), activation='elu', padding='same', name='conv9')(x)
                x = tf.keras.layers.BatchNormalization(name='bn9')(x)
                x = tf.keras.layers.Dropout(0.5, name='dropout9')(x)
                
                # Layer 10
                x = tf.keras.layers.Conv2D(1024, (3,3), activation='elu', padding='same', name='conv10')(x)
                x = tf.keras.layers.BatchNormalization(name='bn10')(x)
                x = tf.keras.layers.MaxPooling2D((2,2), name='pool10')(x)
                
                base_model = Model(inputs=inputs, outputs=x)
            
            
    
    if weights == 'imagenet' and not model_type == 'simplenet':
        base_model.trainable = False
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
    classifier_input = base_model.output
    classifier = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d')(classifier_input)
    for i,neurons in enumerate(classifier_neurons):
        classifier = tf.keras.layers.Dense(neurons, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(l2_reg),name=f'dense{i+1}')(classifier)
        classifier = tf.keras.layers.Dropout(dropout,name=f'clsfr_dropout{i+1}')(classifier)
        
    if classification_type == 'multiclass':
        classifier = tf.keras.layers.Dense(n_classes, activation='softmax',name=f'output')(classifier)
        loss =  tf.keras.losses.CategoricalCrossentropy()
    elif classification_type == 'multilabel':
        classifier = tf.keras.layers.Dense(n_classes, activation='sigmoid',name='output')(classifier)
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    model = Model(inputs=inputs, outputs=classifier, name=model_type)
    
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer == 'lamb':
        optimizer = tfa.optimizers.LAMB(learning_rate=lr)  
    f1_score = tfa.metrics.F1Score(num_classes=n_classes, average="micro",name='f1_score')
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', 'Precision', 'Recall', f1_score])
    if verbose:
        model.summary()
        
    return model
    
    
def train_model(model, train_ds, val_ds, save_path, folder, epochs=100, patience = 10, cbs=[], hps=False, verbose=1,):
    '''
    This function trains the model on the given datasets. It uses the Keras API.
    
    Parameters:
    model: keras.Model/str, the model to train or path to a saved Keras.Model
    train_ds: tf.data.Dataset/Tuple(np.array,np.array), training dataset; either a tf.data.Dataset object or a tuple of (images, labels) as numpy arrays
    val_ds: tf.data.Dataset/ Tuple(np.array,np.array), validation dataset; either a tf.data.Dataset object or a tuple of (images, labels) as numpy arrays
    save_path: str, path to the folder where the model should be saved
    folder: str, name of a subfolder inside save_path where the model should be saved; folder gets created; for example the date and time of training
    epochs: int (optional, default: 100), number of epochs to train the model
    patience: int (optional, default: 10), number of epochs to wait before early stopping
    cbs: list (optional, default: []), list of addidional callbacks to use during training, in addition to the early stopping and model checkpoint
    hps: bool (optional, default: False), whether this function is used for hyperparameter search or normal training
    verbose: int (optional, default: 1), whether to show information during each epoch of training 
    
    Returns:
    for hps=False:
        model: keras.Model, the trained model associated with the best validation f1 score
        history: dict, history of the training process, contains the loss and metrics for each epoch
    for hps=True:
        best_val_f1: float, the best f1 score on the validation set
    '''
    
    model_name = model.name
    savedir = os.path.join(save_path, folder, model_name)
    print('savedir:',savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
        
    checkpoints = os.path.join(save_path, folder, model_name,'best_model_weights')
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints, exist_ok=True)
    
    
    if isinstance(train_ds, tuple):
        images = train_ds[0]
        labels = train_ds[1]
        train_ds = tf.data.Dataset.from_tensor_slices((images,labels))
        train_ds = train_ds.shuffle(len(images))
        train_ds = train_ds.batch(16)
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
    if isinstance(val_ds, tuple):
        images = val_ds[0]
        labels = val_ds[1]
        val_ds = tf.data.Dataset.from_tensor_slices((images,labels))
        val_ds = val_ds.shuffle(len(images))
        val_ds = val_ds.batch(16)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        
    
    
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=patience, verbose=1, mode='max',))
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.001, cooldown=0, min_lr=0))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints+'/checkpoint.weights.h5', monitor='val_f1_score', save_best_only=True, save_weights_only=True, mode='max',))
    for cb in cbs:
        callbacks.append(cb)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=verbose)
    
    print('saved model weights to:',checkpoints)
    #load the best model
    model.load_weights(checkpoints+'/checkpoint.weights.h5')
    #save model as h5 file
    model.save(savedir+'/best_model.h5')
    model.save(savedir+'/best_model', save_format='tf')
    print('saved model to:',savedir+'/best_model')
    
    if hps:
        best_val_f1 = max(history.history['val_f1_score'])
        return best_val_f1
    else:
        return model, history
    
def plot_history(history):
    metrics = [key for key in history.history.keys() if 'val' not in key]
    metrics.remove('lr')
    for metric in metrics:
        plt.figure(figsize=(10,5))
        plt.plot(history.history[metric], label=metric)
        plt.plot(history.history['val_'+metric], label='val_'+metric)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.xticks(range(0,len(history.history[metric])))
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()


def evaluate_model(model, test_ds, classification_type, class_names=None, conf_thresholds=0.5, savedir=None):
    '''
    This function evaluates the model on the given test dataset. It uses the Keras API.
    It calculates the following metrics:
    - accuracy
    - precision
    - recall
    - f1-score
    on the test dataset as a whole and for each class separately.
    It also calculates the micro and macro metrics.
    It saves the metrics in a csv file and creates a bar plot of the classwise metrics.
    It constructs the confusion matrix and saves it as a png file.
    
    
    Parameters:
    model: keras.Model/path to a saved keras.Model, the model to evaluate
    test_ds: tf.data.Dataset, test dataset
    classification_type: str, type of classification; should be 'multiclass' or 'multilabel; multiclass is used when only one class can be present in the image, 
                        multilabel is used when multiple classes can be present in the image'
    class_names: list (optional, default: None), if not provided, the classes are named class0, class1, ...
    conf_thresholds: float/list(optional, default: 0.5), confidence threshold for the predictions; predictions with a confidence above this threshold are positive predictions; can be one value for all classes or a list with a threshold for each class
    savedir: str(optional, default: None), path to the folder where the evaluation results should be saved; saves a confusion matrix, and classification report containing precision, recall, f1-score, and support per class
    '''
    if isinstance(model, str):
        model = load_model(model,custom_objects={'f1_score': tfa.metrics.F1Score})
    img_size = model.input_shape[1:]
    img_size_ds = test_ds.element_spec[0].shape[1:]
    assert img_size == img_size_ds, 'input shape of model and test dataset do not match; model expects input shape {}, test dataset has input shape {}'.format(img_size, img_size_ds)
    assert classification_type in ['multiclass', 'multilabel'], 'classification_type should be one of the following: multiclass, multilabel'
    assert isinstance(model, str) or isinstance(model, tf.keras.Model), 'model should be a path to a saved model or a tf.keras.Model object'
    #create savedir if it does not exist
    if savedir is not None:
        os.makedirs(os.path.join(savedir,'metrics'), exist_ok=True)
    
    x_test, y_test = [],[]
    for x,y in test_ds.as_numpy_iterator():
        x_test.append(x)
        y_test.append(y)
    x_test = np.vstack(x_test)
    y_true = np.vstack(y_test)
    print('shape x_test:',x_test.shape)
    print('shape y_true:',y_true.shape)
    preds = model.predict(x_test)
    metrics = model.evaluate(test_ds, return_dict=True)
    df = pd.DataFrame(metrics, index=[0])
    print('*'*50)
    print('general metrics (with fixed threshold):\n',df.transpose())
    print('*'*50)
    
    if not class_names:
        class_names = [f'class{i}' for i in range(y_true.shape[1])]
    if isinstance(conf_thresholds, float):
        conf_thresholds = np.full(y_true.shape[1], conf_thresholds)
        print('using confidence thresholds {} for all classes'.format(conf_thresholds))
    else:
        assert len(conf_thresholds) == y_true.shape[1], 'conf_thresholds should be a float or a list with the same length as the number of classes; got length {} but should be {}'.format(len(conf_thresholds), y_true.shape[1])
        print('using the following confidence thresholds')
        for i, threshold in enumerate(conf_thresholds):
            print(f'{class_names[i]}: {threshold}')
    
    if classification_type == 'multiclass':
        y_preds = np.argmax(preds, axis=1)
        y_preds = one_hot_encode(y_preds, y_true.shape[1])
    elif classification_type == 'multilabel':
        y_preds = (preds > conf_thresholds).astype(int)
        
    print('shape y_preds:',y_preds.shape)
        
    report = evaluate_predictions(y_preds, y_true, classification_type=classification_type, class_names=class_names)
 
    if savedir is not None:
        report_df = pd.DataFrame(report).transpose()
        if not os.path.exists(os.path.join(savedir,'metrics',model.name)):
            os.makedirs(os.path.join(savedir,'metrics',model.name), exist_ok=True)
        report_df.to_csv(f'{savedir}/metrics/evaluation_report.csv')
    
    create_classwise_bar_plot(report, savedir=f'{savedir}/metrics' if savedir else None)
    
    if classification_type == 'multiclass':
        make_confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_preds, axis=1), class_names, savedir=savedir)
    elif classification_type == 'multilabel':
        nr_combinations = len(class_names)//3 +1
        make_multilabel_confusion_matrix(y_true, y_preds, class_names, nr_combinations=nr_combinations, savedir=savedir)
    
    return report


def make_predictions(model, data, classification_type, img_size=None, conf_thresholds=0.5, get_filenames=False, verbose=1):
    '''
    This function makes predictions on a dataset or single image using the given model. It uses the Keras API.
    The input data can be a tf.data.Dataset object, a image in form of a numpy array, a path to an image or a path to a folder with images.
    The prediction gets made by the model on all images of the data and the predicted labels are returned.
    
    Parameters:
    model: keras.Model/path to a saved keras.Model, the model to make predictions with
    data: tf.data.Dataset/array/image_path/path to a folder with images, test dataset
    classification_type: str, type of classification; should be 'multiclass' or 'multilabel; multiclass is used when only one class can be present in the image, 
                        multilabel is used when multiple classes can be present in the image'
    img_size: tuple (optional, default: None), size to which the images should be resized; if None use original size; should be in the format (height, width, channels)
    conf_thresholds: float/list (optional, default: 0.5), confidence threshold values for the predictions; predictions with a confidence above this threshold are positive predictions; 
                    can be one value for all classes or a list with a threshold for each class; only used for multilabel classification
    get_filenames: bool (optional, default: False), whether to return the filenames of the images used in the prediction
    verbose: int (optional, default: 1), whether to show information during the prediction process
    
    Returns:
    probabilities: numpy array, the predicted probabilities for the data, contains the confidence for each class
    predictions: numpy array, the predicted labels for the data, contains 1 for positive predictions and 0 for negative predictions; is based on the confidence threshold
    '''
    
    if isinstance(model, str):
        model = load_model(model)
    
        # Function to process a single image
    def process_image(img_path):
        try:
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            if verbose > 0:
                print(f"Error processing image {img_path}: {str(e)}")
            return None
    #define a function to read the images from image filepaths
    def parse_images(image_path):
        image = tf.io.read_file(image_path)
        #read image type from the path
        image = tf.image.decode_png(image, channels=img_size[2])
        image = tf.image.resize(image, (img_size[0],img_size[1]),method='area')
        return image
        
    if isinstance(data, list):
        data = np.asarray(data)
        
    if isinstance(data, tf.data.Dataset):
        x_test, y_test =[],[]
        for x,y in data.as_numpy_iterator():
            x_test.append(x)
            y_test.append(y)
        x_test = np.vstack(x_test)
        print('got {} images for prediction'.format(x_test.shape[0]))
        print('shape of images:',x_test.shape)
        probabilities = model.predict(x_test,batch_size=16)
    elif isinstance(data, np.ndarray):
        #add batch dimension if not present
        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)
        print('got {} images for prediction'.format(data.shape[0]))
        probabilities = model.predict(data,batch_size=16)  
    elif isinstance(data, str):
        assert img_size, 'img_size should be provided when using image paths'
        if os.path.isdir(data):
            img_files = [os.path.join(data,img) for img in os.listdir(data) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print('found',len(img_files),'images in the folder')
            ds = tf.data.Dataset.from_tensor_slices(img_files)
            ds = ds.map(parse_images, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(16)
            ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            # img_arrays = [process_image(img) for img in img_files]
            # img_array = np.vstack(img_arrays)
            #print('shape:',img_array.shape)
            probabilities = model.predict(ds,batch_size=16)
            
        elif os.path.isfile(data):
            img_array = process_image(data)
            probabilities = model.predict(img_array)
    else:
        raise ValueError('data should be a tf.data.Dataset, numpy array, image path or folder path')
    
    predictions = np.zeros(probabilities.shape)
    if isinstance(conf_thresholds, float):
        conf_thresholds = np.full(probabilities.shape[1], conf_thresholds)
        conf_thresholds = np.array(conf_thresholds)
    elif len(conf_thresholds) != probabilities.shape[1]:
        raise ValueError(f"conf_thresholds must be a float or a list of {probabilities.shape[1]} floats")
    
    predictions = (probabilities > conf_thresholds).astype(int)
    if classification_type == 'multiclass':
        predictions = np.argmax(probabilities, axis=1)
    if get_filenames:
        return probabilities, predictions, img_files
    return probabilities, predictions
    
def load_model_from_path(model_path):
    '''
    Simple loader function that loads a saved Keras model from a directory and returns it.
    '''
    assert os.path.exists(model_path), f'{model_path} is not a valid path'
    
    model = load_model(model_path, custom_objects={'f1_score': tfa.metrics.F1Score})
    print(f'loaded model {model.name} from {model_path}')
    return model


def optimize_conf_thresholds(model, val_ds, priority_metric='f1_score'):
    '''
    Finds the optimal confidence thresholds for the individual classes for a given model and validation dataset.
    !!Only works for multilabel classification.!!
    It uses the Keras API.
    It performs the following operations:
    - makes predictions on the validation dataset
    - calculates the evaluation metrics for different confidence thresholds
    - finds the optimal confidence threshold for each class
    - creates a bar plot of the optimal confidence thresholds
    
    Parameters:
    model: keras.Model/path to a saved keras.Model, the model to evaluate
    val_ds: tf.data.Dataset, validation dataset
    priority_metric: str (optional, default: 'f1_score'), the metric to optimize for; should be one of the following: 'precision', 'recall', 'f1_score' 
                    (if precision or recall is used, the f1_score is multiplied with the precision or recall to get a combined metric); uses the macro average
    
    Returns:
    optimal_thresholds: list, the optimal confidence thresholds for the individual classes
    '''
    assert priority_metric in ['precision', 'recall', 'f1_score'], 'priority_metric should be one of the following: precision, recall, f1_score'
    if isinstance(model, str):
        model = load_model(model,custom_objects={'f1_score': tfa.metrics.F1Score})
    #get number of classes from the model output shape
    n_classes = model.output_shape[-1]
    
    x_val, y_val = [],[]
    for x,y in val_ds.as_numpy_iterator():
        x_val.append(x)
        y_val.append(y)
    x_val = np.vstack(x_val)
    y_val = np.vstack(y_val)
    probs = model.predict(x_val)
    
    if priority_metric == 'precision':
        print('optimizing for f1-score * precision withweighted average')
    if priority_metric == 'recall':
        print('optimizing for f1-score * recall with weighted average')
    if priority_metric == 'f1_score':
        print('optimizing for f1-score with weighted average')
    
    def objective(trial):
        conf_thresholds =[]
        for i in range(n_classes):
            threshold = trial.suggest_float(f'threshold_{i}', 0.1, 0.9)
            conf_thresholds.append(threshold)
        conf_thresholds = np.array(conf_thresholds)
        y_pred = (probs > conf_thresholds).astype(int)
        report = classification_report(y_val, y_pred, output_dict=True,zero_division=0)
        f1_score = report['weighted avg']['f1-score']
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        
        if priority_metric == 'precision':
            return f1_score* precision**2
        elif priority_metric == 'recall':
            return f1_score* recall**2
        else:
            return f1_score
    
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=300, show_progress_bar=True)
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ", trial.params)
    #make list of optimal thresholds
    optimal_thresholds = []
    for i in range(n_classes):
        optimal_thresholds.append(round(trial.params[f'threshold_{i}'],4))
    
    return optimal_thresholds

wandb_key_ue='0639f18532accef34517cd42931a68968331436e'#access key for weights and biases for ue (intendet to be used by all);
#for login use email: 'tim.ueberlackner@ims-roentgensysteme.de' , passwort: 'WANDBPasswort'
def find_best_model(train_ds, val_ds, test_ds, save_path, classification_type, models='all', model_sizes=['s','m'], max_epochs=60, eval_metric='f1_score',iterations=50, use_wandb=True,wandb_api_key='0639f18532accef34517cd42931a68968331436e',project=None):
    '''
    This function does a Hyperparameter search to find the best model for the given dataset. It uses the Keras API.
    It uses the Optuna library for the Hyperparameter search
    It performs the following operations:
    - defines an objective function for the Hyperparameter search
    - defines the search algorithm and pruner for the Hyperparameter search (based on optuna)
    - selects a set of hyperparameters based on the trial sampler
    - builds a model with the selected hyperparameters
    - trains the model on the training dataset
    - evaluates the model on the validation dataset
    - compare the models based on the evaluation metric with the other models
    - bad models are pruned (the training is stopped early if the model is not performing well)
    - it repeats this process for a given number of iterations
    - the performances are loged with weights and biases and can be viewed in the dashboard
    
    Parameters:
    train_ds: tf.data.Dataset, training dataset
    val_ds: tf.data.Dataset, validation dataset
    test_ds: tf.data.Dataset, test dataset
    save_path: str, path to the folder where the best model should be saved
    classification_type: str, type of classification; should be 'multiclass' or 'multilabel; multiclass is used when only one class can be present in the image, 
                        multilabel is used when multiple classes can be present in the image'
    models: list/str  (optional, default:'all'), list of models to search for; should be one or a combination of the following: 'simplenet','resnet', 'inceptionv3', 'mobilenetv2', 'xception','efficientnet', 'nasnet; if 'all' is used, all models are searched for
    model_sizes: list (optional, default: ['s','m']), list of sizes for the models; should be one or a combination of the following: 's', 'm', 'l'
    max_epochs: int  (optional, default: 60), max number of training epochs per iteration
    eval_metric: str (optional, default: 'f1'), evaluation metric to use for the Hyperparameter search; should be one of the following: 'accuracy', 'precision', 'recall', 'f1'
    iterations: int (optional, default: 50), number of iterations for the Hyperparameter search (should be at least 20)
    use_wandb: bool (optional, default: True), whether to use weights and biases for logging the results
    wandb_api_key: str (optional, default: '0639f18532accef34517cd42931a68968331436e'), the api key for weights and biases; the default key is the key for ue
                    intended to be used by all; for login use 'ue_wandb' or email tim.ueberlackner@ims-roentgensysteme.de with password 'WANDBPasswort'
    project: str (optional, default: None), the name of the project in weights and biases where the results should be logged
    
    The Hyperparameters that are optimized are the following:
        - model_type: the model architecture to use
        - model_size: the size of the model (small, medium, large)
        - learning_rate: the learning rate for the optimizer (one of the most important hyperparameters)
        - optimizer: the optimizer to use for the model
        - classifier_layers: the number of (fully-connected)-layers in the classifier(or head) of the model
        - classifier_units: the number of units in each layer of the classifier
        - dropout: the dropout rate for the classifier (used to prevent overfitting)
        - l2_reg: the l2 regularization parameter for the model (used to prevent overfitting)
        - use_weights: whether to use imagenet weights for the model
        - trainable_layers: the number of layers to make trainable in the pretrained model(only used for transfer learning, if using imagenet weights)
    '''
    assert isinstance(train_ds, tf.data.Dataset), 'train_ds should be a tf.data.Dataset object'
    assert isinstance(val_ds, tf.data.Dataset), 'val_ds should be a tf.data.Dataset object'
    if test_ds:
        assert isinstance(test_ds, tf.data.Dataset), 'test_ds should be a tf.data.Dataset object'
    assert classification_type in ['multiclass', 'multilabel'], 'classification_type should be one of the following: multiclass, multilabel'
    assert eval_metric in ['accuracy', 'precision', 'recall', 'f1_score'], 'eval_metric should be one of the following: accuracy, precision, recall, f1_score'
    #assert iterations >= 20, 'iterations should be at least 20'
    assert isinstance(models, list) or models == 'all', 'models should be a list or all'
    assert models == 'all' or all([model in ['simplenet','resnet', 'inceptionv3', 'mobilenetv2', 'xception','efficientnet','nasnet'] for model in models]), 'models should be one or a combination of the following: simplenet, resnet, inceptionv3, mobilenetv2, xception, efficientnet, nasnet'
    assert isinstance(model_sizes, list) and all([size in ['s','m','l'] for size in model_sizes]), 'model_sizes should be a list of one or a combination of the following: s, m, l'
    assert isinstance(project, str) or project == None, 'project should be a string or None'
    
    project = 'hyperparametersearch_results' if project == None else project
    
    def HPO_objective(trial):
        '''
        Objective function for the Hyperparameter search. It is used to find the best hyperparameters for the model.
        We define it inside the other function so we can use the dataset from the outer function without passing it as an argument.
        It performs the following operations:
        - selects a set of hyperparameters based on the trial sampler
        - builds a model with the selected hyperparameters
        - trains the model on the training dataset
        - evaluates the model on the validation dataset
        - returns the evaluation metric
        
        The Hyperparameters are the following:
        - model_type: the model architecture to use
        - model_size: the size of the model (small, medium, large)
        - learning_rate: the learning rate for the optimizer (one of the most important hyperparameters)
        - optimizer: the optimizer to use for the model
        - classifier_layers: the number of (fully-connected)-layers in the classifier(or head) of the model
        - classifier_units: the number of units in each layer of the classifier
        - dropout: the dropout rate for the classifier (used to prevent overfitting)
        - l2_reg: the l2 regularization parameter for the model (used to prevent overfitting)
        - use_weights: whether to use imagenet weights for the model
        - trainable_layers: the number of layers to make trainable in the pretrained model(only used for transfer learning, if using imagenet weights)
        ''' 
        
        model_type = trial.suggest_categorical('model', compatible_models)
        # Filter model sizes based on compatibility with image size and chosen model
        compatible_sizes = [size for size in model_sizes if model_min_img_sizes[model_type][size] is not None and model_min_img_sizes[model_type][size] <= min_img_size]
        print('compatible_sizes:',compatible_sizes)
        model_size = trial.suggest_categorical('model_size', compatible_sizes)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'lamb'])
        classifier_layers = trial.suggest_int('classifier_layers', 1, 3)
        classifier_units = []
        for i in range(classifier_layers):
            classifier_units.append(trial.suggest_int(f'classifier_units_{i}', 32, 512))
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-1, log=True)
        use_weights = trial.suggest_categorical('use_weights', ['imagenet', None])
        if use_weights == 'imagenet':
            trainable_layers = trial.suggest_int('trainable_layers', 0, 5)
        else:
            trainable_layers = 'all'
        
        
            
        #set up wandb for logging
        hyperparameters = {'model':model_type,
                           'model_size':model_size,
                           'learning_rate':learning_rate,
                           'optimizer':optimizer,
                           'classifier_layers':classifier_layers,
                           'classifier_units':classifier_units,
                           'dropout':dropout,
                           'l2_reg':l2_reg, 
                           'use_weights':use_weights,
                           'trainable_layers':trainable_layers,
                           'classification_type':classification_type}
        print('hyperparameters:',hyperparameters)
        
        if use_wandb:
            if wandb.run is not None:
                print(f'run {wandb.run.name} is running')
                wandb.finish()
            run = wandb.init(project=project, anonymous="allow",mode='online',reinit=True, magic=False, group=str(dt), job_type=model_type, config=hyperparameters,tags=['trial_'+str(trial.number), model_type, classification_type])
            run.name = run.name+f'_trial{trial.number}'
            #run.display(height=720)
        
        #check if image_size is big enough for the model
        min_size = model_min_img_sizes[model_type][model_size]
        if min_size == None:
            return 0
        if input_shape[0] < min_size or input_shape[1] < min_size:
            return 0
        n_classes = train_ds.element_spec[1].shape[-1]
        if input_shape[-1] != 3:
            use_weights = None
        print('n_classes:',n_classes)
        
        model = build_model(input_shape,
                            n_classes, 
                            model_type=model_type, 
                            model_size=model_size, 
                            weights=use_weights, 
                            trainable_layers=trainable_layers, 
                            classifier_neurons=classifier_units,
                            dropout=dropout,
                            l2_reg=l2_reg,
                            optimizer=optimizer,
                            lr=learning_rate,
                            classification_type=classification_type,
                            verbose=0)
        callbacks = [KerasPruningCallback(trial, 'f1_score')]
        if use_wandb:
            callbacks.append(WandbMetricsLogger())
        model, _ = train_model(model,
                               train_ds,
                               val_ds,
                               save_path=save_path,
                               folder='trial'+ str(trial.number)+'_model',
                               epochs=max_epochs,
                               patience = 10,
                               cbs=callbacks,
                               verbose=2)
        
        #save hyperparameters as json file
        with open(os.path.join(save_path,'trial'+ str(trial.number)+'_model',model_type,'hyperparameters.json'), 'w') as f:	
            f.write(json.dumps(hyperparameters)) 
        best_model_path = os.path.join(save_path,'trial'+ str(trial.number)+'_model',model.name,'best_model')
        # run.log_model(best_model_path+'.h5', name = 'best_model')
        # run.log({'best_model': best_model_path+'.h5'})
        if test_ds:
            metrics = model.evaluate(test_ds, return_dict=True)
        else:
            metrics = model.evaluate(val_ds, return_dict=True)
        if eval_metric == 'accuracy':
            score = metrics['accuracy']
        else:
            score = metrics['f1_score']
            if eval_metric == 'precision':
                score *= metrics['precision']
            elif eval_metric == 'recall':
                score *= metrics['recall']
        if use_wandb:
            wandb.log({'eval_score': score})    
            run.finish()
            wandb.finish()
        return score
    
    
    input_shape = train_ds.element_spec[0].shape[1:]
    print('input_shape:',input_shape)
    min_img_size = min(input_shape[:2])
    
    #define the hyperparameters to search for
    model_types = ['simplenet','resnet', 'inceptionv3', 'mobilenetv2', 'xception','efficientnet','nasnet'] if models == 'all' else models
    compatible_models = [model for model in model_types if any(model_min_img_sizes[model][size] <= min_img_size for size in model_sizes)]
    print('compatible_models for img size:',compatible_models)
    if not compatible_models:
        raise ValueError(f"No compatible models found for image size {min_img_size}; consider resizing the images to at least 32x32 or to 75x75 to use all models")
    
    
    if use_wandb:
        os.environ["WANDB_MODE"] = "online"
        if wandb_api_key==None:
            os.environ["WANDB_API_KEY"] = ''
            wandb.login(anonymous='must', relogin=True, force=True)
        else:
            os.environ["WANDB_API_KEY"] = wandb_api_key
            wandb.login(key=wandb_api_key)
        
        
    dt= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.HyperbandPruner())
    study.optimize(HPO_objective, n_trials=iterations)
    
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_intermediate_values(study)
    optuna.visualization.plot_parallel_coordinate(study)
    optuna.visualization.plot_param_importances(study)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    print("Number of finished trials: {}".format(len(study.trials)))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial
    best_params = trial.params
    best_score = trial.value
    print("  Value: {}".format(best_score))
    print("  Params: ", best_params)
    print('with model:',trial.params['model'])
    #copy contents from the best trial to a new folder named best_model
    os.makedirs(os.path.join(save_path,'best_trial'), exist_ok=True)
    shutil.copytree(os.path.join(save_path,'trial'+ str(trial.number)+'_model'), os.path.join(save_path,'best_trial','trial'+ str(trial.number)), dirs_exist_ok=True)
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    #savepath of best model
    best_save_path = os.path.join(save_path,'trial'+ str(trial.number)+'_model',trial.params['model'],'best_model')
    
    return best_params, best_score, best_save_path
    

def cross_validation(model, img_path, label_path, save_path, n_classes, classification_type, folds=5, epochs = 50, augment_strength=0.1, img_type='png',conf_thresholds=0.5, verbose=1, save_file_splits=False):
    '''
    Perform cross validation on the given dataset. It uses the Keras API.
    It performs the following operations:
    - reads the images and labels from the given paths
    - constructs the dataset
    - splits the dataset into the given number of folds
    - trains the model on the training folds
    - evaluates the model on the validation fold
    - repeats this process for each fold
    - returns the mean metrics over all folds
    
    Parameters:
    model: keras.Model/path to a saved keras.Model, the model to train
    img_path: str, path to the folder that contains the images; images should be in .png or .jpeg format
    label_path: str, path to the file that contains the labels; labels should be in .csv format where the first column contains the image names and the other columns contain the labels for each class
    save_path: str, path to the folder where the best model should be saved
    n_classes: int, number of classes
    classification_type: str, type of classification; should be 'multiclass' or 'multilabel; multiclass is used when only one class can be present in the image,
    folds: int (optional, default: 5), number of folds for the cross validation; standard is 5
    epochs: int (optional, default: 50), number of training epochs in each fold 
    augment_strength: float(optional, default: 0.1), strength of the augmentation; should be between 0 and 1
    img_type: str(optional, default: 'png'), type of the images; should be 'png' or 'jpeg'
    conf_thresholds: float/list (optional, default: None), confidence threshold values for the predictions; predictions with a confidence above this threshold are positive predictions;
    verbose: int, whether to show information during the training process
    save_file_splits: bool(optional, default: False), whether to save which images are in which fold; saves the image names to a text file for each fold,
                        these are saved in the folder 'splits' in the save_path
    
    Returns:
    mean_metrics: dict, the mean metrics over all folds
    
    '''
    assert os.path.exists(img_path), 'image_path does not exist'
    assert os.path.exists(label_path), 'label_path does not exist'
    assert img_type in ['png', 'jpeg'], f'img_type should be one of the following: png, jpeg, got {img_type}'
    assert isinstance(model, str) or isinstance(model, tf.keras.Model), 'model should be a path to a saved model or a tf.keras.Model object'
    #load model if path is given
    if isinstance(model, str):
        model = load_model(model,custom_objects={'f1_score': tfa.metrics.F1Score})
    #get image size from model input shape
    img_size = model.input_shape[1:]
    print('img_size:',img_size)
        
    labels = pd.read_csv(label_path)
    img_files = labels.iloc[:,0].values 
    img_files = img_path + '\\' + img_files #make whole path for images
    labels = labels.drop(labels.columns[0], axis=1)
    class_names = list(labels.columns)
    labels = labels.to_numpy()
    assert len(img_files) == len(labels), 'number of images and labels do not match'
    image_of_each_class ={}
    while len(image_of_each_class) < n_classes:
        for i in range(len(labels)):
            label = np.argmax(labels[i])
            if label not in image_of_each_class.keys():
                image_of_each_class[label] = img_files[i]
    save_path = save_path + '/' + model.name
    datasets = create_cross_validation_datasets(img_files, labels, n_folds=folds, img_size=img_size, img_type=img_type,save_file_splits=save_file_splits,splits_path=save_path+'/splits')
    
    @tf.autograph.experimental.do_not_convert
    def augment_image(image, label):
        return trainAug(image, training=True), label
    
    #define a object containing augmentation layers; is used for augmenting the images in the training dataset
    trainAug = Sequential([tf.keras.layers.RandomFlip("horizontal_and_vertical"),
                           #tf.keras.layers.RandomTranslation(height_factor=augment_strength, width_factor=augment_strength, fill_mode='reflect',interpolation="bilinear"),#translation by % of the image size, not applicable for all datasets, uncomment if needed	
                            tf.keras.layers.RandomContrast(augment_strength),
                            tf.keras.layers.RandomBrightness(augment_strength),
                            tf.keras.layers.GaussianNoise(augment_strength),
                            tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0, 255)),#clip values to [0,1]
                            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.uint8)),#cast to int8
                            ])
    #perform cross validation
    fold_reports = []
    for i in range(folds):
        print('fold:',i)
        
        # Create validation dataset
        val_ds = datasets[i]
        val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)  
        # Create training dataset
        train_ds = None
        for j in range(folds):
            if j != i:
                if train_ds is None:
                    train_ds = datasets[j]
                else:
                    train_ds = train_ds.concatenate(datasets[j])
        # Apply augmentation to training dataset
        train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)

        model, history = train_model(model, train_ds, val_ds, save_path=save_path, folder=f'fold_{i}', epochs=epochs, patience = 10, hps=False, verbose=verbose)
        report = evaluate_model(model, val_ds, class_names=class_names, classification_type=classification_type, savedir=save_path+'/fold_'+str(i),conf_thresholds=conf_thresholds)
        print(report.keys())
        fold_reports.append({'fold':i,'report':report})
    
    df_list = []

    for item in fold_reports:
        fold = item['fold']
        report = item['report']
        
        # Convert the report to a DataFrame
        df = pd.DataFrame(report).transpose()
        
        # Add 'fold' and 'class' columns
        df['fold'] = fold
        df['class'] = df.index
        
        # Reset index to make 'class' a regular column
        df = df.reset_index(drop=True)
        
        # Reorder columns to put 'fold' first and 'class' second
        columns = ['fold', 'class'] + [col for col in df.columns if col not in ['fold', 'class']]
        df = df[columns]
        
        df_list.append(df)

    # Concatenate all DataFrames
    final_df = pd.concat(df_list, ignore_index=True)
    print('foldwise and classwise metrics:\n',final_df)
    os.makedirs(save_path+'/metrics', exist_ok=True)
    final_df.to_csv(save_path+'/metrics/class_metrics.csv')
    mean_metrics = final_df.groupby(final_df['class']).mean()
    mean_metrics.drop(columns='fold', inplace=True)
    print('mean metrics:\n',mean_metrics)
    os.makedirs(save_path+'/avg_metrics', exist_ok=True)
    mean_metrics.to_csv(save_path+'/avg_metrics/mean_metrics.csv')
    return mean_metrics
    
    
def create_cross_validation_datasets(img_files, labels, n_folds, img_size, img_type='png',save_file_splits=False,splits_path=None):
    #define a function to read the images from image filepaths
    @tf.autograph.experimental.do_not_convert
    def parse_images(image_path, label):
        image = tf.io.read_file(image_path)
        if img_type == 'png':
            image = tf.image.decode_png(image, channels=img_size[2])
        elif img_type == 'jpeg':
            image = tf.image.decode_jpeg(image, channels=img_size[2])
        image = tf.image.resize(image, (img_size[0],img_size[1]),method='area')
        return image, label
    
    dataset = tf.data.Dataset.from_tensor_slices((img_files, labels))
    dataset = dataset.shuffle(len(img_files), reshuffle_each_iteration=False, seed=42)
    #dataset = dataset.map(parse_images, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Split the dataset into n_folds
    fold_size = len(img_files) // n_folds
    datasets = [dataset.skip(i * fold_size).take(fold_size) for i in range(n_folds)]
    
    if save_file_splits==True:
        print('saving file splits in:',splits_path)
        for i,fold in enumerate(datasets):
            #get the image files and labels
            img_files = []
            labels = []
            for img, label in fold:
                full_filename = img.numpy().decode('utf-8')
                base_filename = os.path.basename(full_filename)
                img_files.append(base_filename)
                labels.append(label.numpy())
            #save the image files and labels
            #create a text file with the image files
            os.makedirs(splits_path, exist_ok=True)
            with open(os.path.join(splits_path,f'fold_{i}_img_files.txt'), 'w') as f:
                for item in img_files:
                    item = item.split('\\')[-1]
                    f.write("%s\n" % item)
                    
    # Now apply parse_images to each fold
    datasets = [fold.map(parse_images, num_parallel_calls=tf.data.AUTOTUNE) for fold in datasets]
    
    return datasets
     

def make_confusion_matrix(y_true, y_pred, class_names, savedir=None):   
    '''
    Helper function to create a confusion matrix for a multiclass classification problem.
    It uses the sklearn library to calculate the confusion matrix and the seaborn library to plot the confusion matrix.
    Parameters:
    y_true: numpy array, true labels
    y_pred: numpy array, predicted labels
    class_names: list, list of class names
    savedir: str, path to the folder where the confusion matrix should be saved
    '''
    cm = confusion_matrix(y_true, y_pred)
    #display the confusion matrix with seaborn
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket_r',linewidth=.5,square=True, xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if savedir is not None:
        plt.savefig(f'{savedir}/metrics/confusion_matrix.png')
    plt.show()
    


def make_multilabel_confusion_matrix(y_true, y_pred, class_names, nr_combinations=5, savedir=None):
    """
    Helper function to create a confusion matrix for a multilabel classification problem.
    It uses the sklearn library to calculate the confusion matrix and the seaborn library to plot the confusion matrix.
    It displays the individual classes and the most common combinations of classes.
    Parameters:
    y_true: numpy array, true labels
    y_pred: numpy array, predicted labels
    class_names: list, list of class names
    nr_combinations: int, number of combinations to display
    savedir: str, path to the folder where the confusion matrix should be saved
    """
    pred_dict ={class_names[i]: {'true_pos': 0, 'false_pos': 0, 'false_neg': 0} for i in range(len(class_names))}
    label_dict = {}
    pred_labels_dict ={}
    for i in range(len(y_pred)):
        true = np.where(y_true[i] == 1)[0]
        preds = np.where(y_pred[i] == 1)[0]
        true_string = convert_label_to_string(true, class_names)
        preds_string = convert_label_to_string(preds, class_names)
        label_dict.setdefault(true_string, {'count': 0, 'preds': {}})
        label_dict[true_string]['count'] += 1
        label_dict[true_string]['preds'].setdefault(preds_string, 0)
        label_dict[true_string]['preds'][preds_string] += 1
        pred_labels_dict.setdefault(preds_string, 0)
        pred_labels_dict[preds_string] += 1

        for i in range(len(preds)):
            if preds[i] in true:
                pred_dict[class_names[preds[i]]]['true_pos'] += 1
            else:
                pred_dict[class_names[preds[i]]]['false_pos'] += 1
        for i in range(len(true)):
            if true[i] not in preds:
                pred_dict[class_names[true[i]]]['false_neg'] += 1
                
    #all combinations of classes
    true_labels = list(label_dict.keys())
    #remove class_names from true_labels
    new_true_labels = [label for label in true_labels if label not in class_names]
    print('new true labels:', new_true_labels)
    true_labels_count = [label_dict[key]['count'] for key in new_true_labels]
    # Sorting the true labels by their count in descending order and selecting top 5
    top_true_indices = np.argsort(true_labels_count)[::-1][:nr_combinations]
    top_true_labels = np.array(new_true_labels)[top_true_indices]
    #print('top true labels:', top_true_labels)
    #sort top true labels in alphabetical order
    sorted_true_labels = sorted(top_true_labels, key=custom_sort)
    print('sorted true labels:', sorted_true_labels,'type:', type(sorted_true_labels))
    #add the class names to the front of the list sorted_true_labels
    #class_names to list
    class_names = list(class_names)
    sorted_true_labels = class_names + sorted_true_labels

    pred_labels = list(pred_labels_dict.keys())
    new_pred_labels = [label for label in pred_labels if label not in class_names]
    pred_labels_count = [pred_labels_dict[key] for key in new_pred_labels]
    top_pred_indices = np.argsort(pred_labels_count)[::-1][:nr_combinations]
    top_pred_labels = np.array(new_pred_labels)[top_pred_indices]
    #print('top pred labels:', top_pred_labels)
    #sort top pred labels in alphabetical order
    sorted_pred_labels = sorted(top_pred_labels, key=custom_sort)
    print('sorted pred labels:', sorted_pred_labels)
    
    sorted_pred_labels = class_names + sorted_pred_labels
    #print('sorted pred labels:', sorted_pred_labels)

    #make a confusion matrix with only the top 10 labels
    cm = np.zeros((len(sorted_true_labels ),len(sorted_pred_labels )))
    for i in range(len(sorted_true_labels)):
        for j in range(len(sorted_pred_labels)):
            #print('true label:', sorted_true_labels[i],'pred label:', sorted_pred_labels[j])
            # Check if the true label exists in label_dict and has preds for the current pred label
            if sorted_true_labels[i] in label_dict and sorted_pred_labels[j] in label_dict[sorted_true_labels[i]]['preds']:
                cm[i,j] = label_dict[sorted_true_labels[i]]['preds'].get(sorted_pred_labels[j], 0)
            else:
                cm[i,j] = 0  # Assign 0 if the condition is not met
                
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt='g',xticklabels=sorted_pred_labels, yticklabels=sorted_true_labels,linewidth=.5,square=True, cmap='rocket_r')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if savedir is not None:
        plt.savefig(f'{savedir}/metrics/multilabel_confusion_matrix.png')
    plt.show()

def convert_label_to_string(idx, class_names):
    if len(idx) == 0:
        class_string = 'None'
    elif len(idx) == 1:
        class_string = class_names[idx[0]]
    else:
        class_string = '+'.join(class_names[j] for j in idx)
        #remove all instances of 'class' except the first one
        class_string = class_string.replace('class', '')
        class_string = class_string.replace('_', '+')
        class_string ='class'+ class_string
    return class_string

# Custom sorting function, adjusted for tuples from enumerate
def custom_sort(label):
    #label = item[1]  # Adjusted to get the label from the tuple
    plus_count = label.count('+')
    return (plus_count, label)
    
    
def get_dataset_composition(labels, class_names,set_name=''):
    class_dict = {}
    pos_labels =0
    for label in labels:
        idx = np.where(label == 1)[0]
        #join string in form class1+2+3
        class_string = '+'.join(class_names[j] for j in idx)    
        for id in idx:
            pos_labels += 1
            if class_names[id] in class_dict:
                class_dict[class_names[id]] += 1
            else:
                class_dict[class_names[id]] = 1     
        
    print(f'class distribution {set_name}:')
    #sort class_dict for alphabetical order of keys
    class_dict = dict(sorted(class_dict.items(), key=lambda item: item[0]))
    classes = []
    percentages = []
    for key, value in class_dict.items():
        print('samples for class {:<15}: {:<6} {:.1%}'.format(key, value, value/pos_labels))
        classes.append(key)
        percentages.append(round(value/pos_labels *100,2))
    print('\n')
    #make pie chart of class distribution
    create_pie_chart(classes, percentages,f'class distribution of {set_name}')
    
    return class_dict


def create_pie_chart(classes, percentages,title):
    # Creating a DataFrame
    df = pd.DataFrame({'Class': classes, 'Percentage': percentages})
    # Adding a new column for labeling: Classes above 1% retain their name; others are grouped into "Other"
    df['Label'] = df.apply(lambda row: row['Class'] if row['Percentage'] > 0.38 else 'Other', axis=1)
    # Grouping the "Other" categories together by summing their percentages
    df_grouped = df.groupby('Label', as_index=False).agg({'Percentage': 'sum'})
    sorted_index = sorted(df_grouped.index, key=lambda idx: custom_sort_key(df_grouped.loc[idx]))
    # Reindex the DataFrame based on the sorted index
    df_grouped_sorted = df_grouped.reindex(sorted_index)
    #reindex
    df_grouped_sorted.reset_index(drop=True, inplace=True)
    #print(df_grouped_sorted)
    num_labels = df_grouped_sorted['Label'].nunique()  # Number of unique labels
    deep_palette = sns.color_palette("deep", n_colors=num_labels)
    deep_palette_hex = [mcolors.rgb2hex(color) for color in deep_palette]

    # Creating the pie chart with Plotly Express
    #use the "deep" color palette from seaborn
    fig = px.pie(df_grouped_sorted, names='Label', values='Percentage',color_discrete_sequence=deep_palette_hex)  # The 'hole' parameter creates a donut-like pie chart
    fig.update_layout(width=500, height=500)

    fig.update_traces(textinfo='label+percent',textposition='inside')#,texttemplate='%{label}:\n %{percent:.1%}')
    #fig.update_traces(texttemplate='%{label}', textposition='outside',)
    #fig.update_traces(texttemplate='%{percent:.1%}', textposition='inside',)
    #pio.write_image(fig, r'C:\Users\timue\Desktop\Mastrarbeit\Bilder\class_distribution.pdf')
    #add title
    fig.update_layout(title_text=title, title_x=0.5, title_y=0.95)
    fig.show()
    
def custom_sort_key(row):
        # Special treatment for "Other"
        if row['Label'] == 'Other':
            return (float('inf'), row['Label'])  # Ensure "Other" is always last, but also maintain alphabetical sorting just in case
        # First element of the tuple is the count of "+" symbols, and the second is the label itself for alphabetical sorting
        return (row['Label'].count("+"), row['Label'])
    
def export_onnx_model(model,save_path=None):
    '''
    Export a keras model to onnx format
    Parameters:
    model: keras.Model/str, the model to export or the path to the saved model
    save_path: (optional, default: None) str, path where the onnx model should be saved, If not given, a default path will be created
    
    !! If error during saving, try starting VS Code as administrator !!
    
    If model is a string (path to saved model):
       - Loads the model from the given path.
       - If save_path is None, creates an 'onnx_model' folder in the same directory as the input model
         and saves the ONNX model there as 'model.onnx'.
       - otherwise saves it to save_path
    If model is a Keras model object:
       - If save_path is None, creates an 'onnx_models' folder in the current directory,
         then creates a subfolder with the model's name and saves the ONNX model there.
       - otherwise saves it to save_path
    '''
    if isinstance(model, str):
        model_path = model
        model = load_model(model,custom_objects={'f1_score': tfa.metrics.F1Score})
        if save_path is None:
            #create path one level above the model folder
            save_path = os.path.join(os.path.dirname(model_path), 'onnx_model')
            os.makedirs(save_path, exist_ok=True)

    elif save_path is None:
        #create path model folder
        save_path = 'onnx_models'
        save_path = os.path.join(save_path,model.name)
        os.makedirs(save_path, exist_ok=True)
        
    if not os.path.exists(save_path):
        save_path = save_path + '/' + model.name
        os.makedirs(save_path, exist_ok=True)
    full_path = os.path.abspath(os.path.join(save_path, 'model.onnx'))
    print('saving onnx model to:',os.path.dirname(full_path))
    onnx_model = onnxmltools.convert_keras(model)
    onnxmltools.utils.save_model(onnx_model, full_path)
    
    
    

def multimodal_prediction_batch(model1,model2, data1, data2, n_classes, classification_type, img_size1=None, img_size2=None, conf_thresholds=0.5, class_names=None, T=0.1):
    '''
    This function combines the predictions of two modalities on a whole batch of data. It uses a individual model for each modality. 
    All detected classes of the whole batch and of both modalities are combined into a single prediction.
    If you would like to combine only the predictions of individual images for both modalities use the multimodal_prediction_single() -function.
    The Modalities can have different number of classes But classes must overlap, but classes must overlap.. For example: modality1: [0,1,2,3], modality2: [0,1,2,4,5,6]. 
    The classes that are present in both modalities are combined. A class that is only present in one modality is used alone. 
    
    Parameters:
    model1: keras.Model/str, the model for the first modality or the path to the saved model
    model2: keras.Model/str, the model for the second modality or the path to the saved model
    data1: numpy array/str, data for the first modality or the path to the data
    data2: numpy array/str, data for the second modality or the path to the data
    n_classes: int, number of classes
    classification_type: str, type of classification problem; should be one of the following: 'multiclass', 'multilabel'
    img_size1: tuple, the size of the images for the first modality
    img_size2: tuple, the size of the images for the second modality
    conf_thresholds: float/np.array, the confidence threshold for the predictions; if a float is given, the same threshold is used for all classes
    class_names: list, list of class names
    T: float/list, the temperature parameter for the softmax function; if big all predictions are equally weighted, if 0 the highest prediction will turn to 1 and the others to 0, if 1 its the standard softmax function 
        if a float is given, the same value is used for all classes
    
    Returns:
    class_preds: list, the predicted classes of the combined modalities
    class_probs: list, the probabilities for each class of the combined modalities
    class_preds1: list, the predicted classes of the first modality
    class_preds2: list, the predicted classes of the second modality
    '''
    if isinstance(model1, str):
        model1 = load_model(model1,custom_objects={'f1_score': tfa.metrics.F1Score})
    if isinstance(model2, str):
        model2 = load_model(model2,custom_objects={'f1_score': tfa.metrics.F1Score})
    if isinstance(conf_thresholds, float):
        conf_thresholds = np.full(n_classes, conf_thresholds)
        conf_thresholds = np.array(conf_thresholds)
    else:
        conf_thresholds=np.array(conf_thresholds)
    if isinstance(T, float) or isinstance(T, int):
        T = np.full(n_classes, T)
        T = np.array(T)
    else:
        T = np.array(T)
    
    assert len(conf_thresholds)==n_classes, 'nr of thresholds must be the same as nr of classes'
    assert len(T)==n_classes, 'nr of temperature parameters must be the same as nr of classes'
        
    if class_names is not None:
        assert len(class_names)==len(conf_thresholds)==n_classes, f'n_classes nr of thresholds and nr of class_names must be the same. got {n_classes} classes, {len(conf_thresholds)} thresholds and {len(class_names)} class names'
    
    print('making prediction for modality 1')
    probs1, preds1 = make_predictions(model1, data1, img_size=img_size1, classification_type=classification_type)
    print('making prediction for modality 2')
    probs2, preds2 = make_predictions(model2, data2, img_size=img_size2, classification_type=classification_type)
    #since the number of classes and number of images of the two modalities can be different, we fill up the missing values with nan values
    probs = np.full((max(probs1.shape[0],probs2.shape[0]),max(probs1.shape[1],probs2.shape[1])), np.nan)#make a np array filled with nan values
    probs[:probs1.shape[0],:probs1.shape[1]] = probs1
    probs1 = probs
    probs[:probs2.shape[0],:probs2.shape[1]] = probs2
    probs2 = probs

    
    class_probs = []# contains the probabilities for each class
    class_preds=[]# contains the predicted classes of the combined modalities
    class_preds1=[]# contains the predicted classes of modality1
    class_preds2=[]# contains the predicted classes of modality2
    for i in range(n_classes):
        threshold = conf_thresholds[i]
        mean_prob, mean_prob1, mean_prob2 = combine_predictions(probs1[:,i], probs2[:,i], T[i])
        class_probs.append(mean_prob)
        if mean_prob > threshold:
            class_preds.append(i)
        if mean_prob1 >threshold:
            class_preds1.append(i)
        if mean_prob2 >threshold:
            class_preds2.append(i)
        
    return class_preds, class_probs, class_preds1, class_preds2


def multimodal_prediction_single(model1,model2, data1, data2, n_classes, classification_type, img_size1=None, img_size2=None, conf_thresholds=0.5, class_names=None, T=0.1):
    assert len(data1)== len(data2), 'data of modality 1 must have the same length of data of modality 2'
    if isinstance(conf_thresholds, float):
        conf_thresholds = np.full(n_classes, conf_thresholds)
        conf_thresholds = np.array(conf_thresholds)
    else:
        conf_thresholds=np.array(conf_thresholds)
    if isinstance(T, float) or isinstance(T, int):
        T = np.full(n_classes, T)
        T = np.array(T)
    else:
        T = np.array(T)
        
    if class_names is not None:
        assert len(class_names)==len(conf_thresholds)==n_classes, f'n_classes nr of thresholds and nr of class_names must be the same. got {n_classes} classes, {len(conf_thresholds)} thresholds and {len(class_names)} class names'

    print('making predictions for modality 1')
    probs1, preds1 = make_predictions(model1, data1, img_size=img_size1, classification_type=classification_type)
    print('making predictions for modality 2')
    probs2, preds2 = make_predictions(model2, data2, img_size=img_size2, classification_type=classification_type)
    #since the number of classes and number of images of the two modalities can be different, we fill up the missing values with nan values
    probs = np.full((max(probs1.shape[0],probs2.shape[0]),max(probs1.shape[1],probs2.shape[1])), np.nan)#make a np array filled with nan values
    probs[:probs1.shape[0],:probs1.shape[1]] = probs1
    probs1 = probs
    probs[:probs2.shape[0],:probs2.shape[1]] = probs2
    probs2 = probs
    
    class_probs = []# contains the probabilities for each class
    class_preds=[]# contains the predicted classes of the combined modalities
    class_preds1=[]# contains the predicted classes of modality1
    class_preds2=[]# contains the predicted classes of modality2
    for prob1, prob2 in zip(probs1,probs2):
        all_probs=[]
        all_probs1=[]
        all_probs2=[]
        preds = []
        preds1 = []
        preds2 = []
        for i in range(n_classes):
            threshold = conf_thresholds[i]
            mean_prob, mean_prob1, mean_prob2 = combine_predictions(prob1[i], prob2[i], T[i])
            all_probs.append(mean_prob)
            all_probs1.append(mean_prob1)
            all_probs2.append(mean_prob2)
            if classification_type == 'multilabel':  
                if mean_prob > threshold:
                    preds.append(i)
                if mean_prob1 >threshold:
                    preds1.append(i)
                if mean_prob2 >threshold:
                    preds2.append(i)
        if classification_type == 'multiclass':
            preds = np.argmax(all_probs)
            preds1 = np.argmax(all_probs1)
            preds2 = np.argmax(all_probs2)
                
        class_probs.append(all_probs)
        class_preds.append(preds)
        class_preds1.append(preds1)
        class_preds2.append(preds2)
        
    return class_preds, class_probs, class_preds1, class_preds2
 
 
def combine_predictions(probs1, probs2, T=0.1):
    #filter out nan values
    probs1 = probs1[~np.isnan(probs1)]
    probs2 = probs2[~np.isnan(probs2)]
    # use the confidence of the predictions as weights for the sum
    softmax1 = np.exp(probs1/T) / np.sum(np.exp(probs1/T))
    mean_prob1 = np.average(probs1, weights=softmax1)
    mean_prob1 = round(mean_prob1,4)
    
    # use the confidence of the predictions as weights for the sum
    softmax2 = np.exp(probs2/T) / np.sum(np.exp(probs2/T))
    mean_prob2 = np.average(probs2, weights=softmax2)
    mean_prob2 = round(mean_prob2, 4)
    
    #combine the predictions of the two modalities using the softmax fusion algorithm
    softmax = np.exp(np.hstack((mean_prob1, mean_prob2))/T) / np.sum(np.exp(np.hstack((mean_prob1, mean_prob2))/T))
    mean_prob = np.average(np.hstack((mean_prob1, mean_prob2)), weights=softmax)
    mean_prob = round(mean_prob, 4)
    
    return mean_prob, mean_prob1, mean_prob2
    
def optimize_multimodal_parameters(model1, model2, data1, data2, labels, n_classes, classification_type, type='batch',img_size1=None, img_size2=None):
    '''
    Optimizes the parameters of the multimodal fusion algorithm using Optuna. 
    The parameters that can be optimized are the temperature parameter T and the confidence thresholds for the predictions(only applicable for multilabel classification).
    The function can be used for both single and batch predictions. For batch predictions, the data must be given as a list of numpy arrays.
    The function uses the weighted average f1-score as the objective function.
    
    Parameters:
    model1: keras.Model/str, the model for the first modality or the path to the saved model
    model2: keras.Model/str, the model for the second modality or the path to the saved model
    data1: numpy array/str, data for the first modality or the path to the data
    data2: numpy array/str, data for the second modality or the path to the data
    labels: numpy array, the labels
    n_classes: int, number of classes
    classification_type: str, type of classification problem; should be one of the following: 'multiclass', 'multilabel'
    type: str, type of prediction; should be one of the following: 'batch', 'single'
    
    returns:
    best_params: dict, the best parameters found by the optimization
    
    
    '''
    assert type in ['batch', 'single'], 'type must be either batch or single'
    
    def batch_objective(trial):
        Ts = []
        conf_thresholds = []
        for i in range(n_classes):
            if classification_type == 'multiclass':
                T = trial.suggest_float(f'T_{i}', 0.01, 10, log=True)
                Ts.append(T)
            elif classification_type == 'multilabel':
                T = trial.suggest_float(f'T_{i}', 0.01, 10, log=True)
                conf_threshold = trial.suggest_float(f'conf_threshold_{i}', 0, 1)
                Ts.append(T)
                conf_thresholds.append(conf_threshold)
                
        class_preds=[]# contains the predicted classes of the combined modalities
        for prob1, prob2 in zip(batch_probs1, batch_probs2):
            all_probs=[]
            preds = []
            for i in range(n_classes):
                mean_prob, mean_prob1, mean_prob2 = combine_predictions(prob1[:,i], prob2[:,i], Ts[i])
                all_probs.append(mean_prob)
                if classification_type == 'multilabel':  
                    threshold = conf_thresholds[i]
                    if mean_prob > threshold:
                        preds.append(i)
            if classification_type == 'multiclass':
                preds = np.argmax(all_probs)
            class_preds.append(preds)
            
        predictions = one_hot_encode(class_preds, 10, classification_type=classification_type)
        assert len(predictions) == len(labels), 'number of predictions and labels must be the same'
        assert len(predictions[0]) == len(labels[0]), 'number of predictions and labels must be the same'
        assert len(labels[0]) == n_classes, 'n_classes must be the same as the number of classes in the labels'
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        f1 = report['weighted avg']['f1-score']
        return f1
    
    def single_objective(trial):
        Ts = []
        conf_thresholds = []
        for i in range(n_classes):
            if classification_type == 'multiclass':
                T = trial.suggest_float(f'T_{i}', 0.01, 10, log=True)
                Ts.append(T)
            elif classification_type == 'multilabel':
                T = trial.suggest_float(f'T_{i}', 0.01, 10, log=True)
                conf_threshold = trial.suggest_float(f'conf_threshold_{i}', 0, 1)
                Ts.append(T)
                conf_thresholds.append(conf_threshold)
                
        class_preds=[]# contains the predicted classes of the combined modalities
        for prob1, prob2 in zip(probs1,probs2):
            all_probs=[]
            preds = []
            for i in range(n_classes):
                mean_prob, mean_prob1, mean_prob2 = combine_predictions(prob1[i], prob2[i], Ts[i])
                all_probs.append(mean_prob)
                if classification_type == 'multilabel':  
                    threshold = conf_thresholds[i]
                    if mean_prob > threshold:
                        preds.append(i)
            if classification_type == 'multiclass':
                preds = np.argmax(all_probs)
            class_preds.append(preds)
            
        predictions = one_hot_encode(class_preds, 10, classification_type=classification_type)
        assert len(predictions) == len(labels), f'number of predictions and labels must be the same; got {len(predictions)} predictions and {len(labels)} labels'
        assert len(predictions[0]) == len(labels[0]), 'number of predictions and labels must be the same; got {len(predictions[0])} predictions and {len(labels[0])} labels'
        assert len(labels[0]) == n_classes, 'n_classes must be the same as the number of classes in the labels; got {len(labels[0])} classes and {n_classes} n_classes'
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        f1 = report['weighted avg']['f1-score']
        return f1
                
    if type == 'batch':
        batch_probs1 = []
        batch_probs2 = []
        for batch1, batch2 in zip(data1,data2):
            probs1, preds1 = make_predictions(model1, batch1, img_size=img_size1, classification_type=classification_type)
            probs2, preds2 = make_predictions(model2, batch2, img_size=img_size2, classification_type=classification_type)
            #since the number of classes and number of images of the two modalities can be different, we fill up the missing values with nan values
            probs = np.full((max(probs1.shape[0],probs2.shape[0]),max(probs1.shape[1],probs2.shape[1])), np.nan)#make a np array filled with nan values
            probs[:probs1.shape[0],:probs1.shape[1]] = probs1
            probs1 = probs
            probs[:probs2.shape[0],:probs2.shape[1]] = probs2
            probs2 = probs
            batch_probs1.append(probs1)
            batch_probs2.append(probs2)
        
        study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
        study.optimize(batch_objective, n_trials=300)
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ", trial.params)
        if classification_type == 'multiclass':
            optimal_T = []
            for i in range(n_classes):
                optimal_T.append(trial.params[f'T_{i}'])
            optimal_thresholds = []
        elif classification_type == 'multilabel':
            optimal_T = []
            optimal_thresholds = []
            for i in range(n_classes):
                optimal_T.append(trial.params[f'T_{i}'])
                optimal_thresholds.append(trial.params[f'conf_threshold_{i}'])
        return optimal_T, optimal_thresholds
            
    elif type == 'single':
        print('making predictions for modality 1')
        probs1, preds1 = make_predictions(model1, data1, img_size=img_size1, classification_type=classification_type)
        print('making predictions for modality 2')
        probs2, preds2 = make_predictions(model2, data2, img_size=img_size2, classification_type=classification_type)
        #since the number of classes and number of images of the two modalities can be different, we fill up the missing values with nan values
        probs = np.full((max(probs1.shape[0],probs2.shape[0]),max(probs1.shape[1],probs2.shape[1])), np.nan)#make a np array filled with nan values
        probs[:probs1.shape[0],:probs1.shape[1]] = probs1
        probs1 = probs
        probs[:probs2.shape[0],:probs2.shape[1]] = probs2
        probs2 = probs
        
        study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
        study.optimize(single_objective, n_trials=300)
        print("Best trial:")
        trial = study.best_trial
        print("  Value: {}".format(trial.value))
        print("  Params: ", trial.params)
        
        if classification_type == 'multiclass':
            optimal_T = []
            for i in range(n_classes):
                optimal_T.append(trial.params[f'T_{i}'])
            optimal_thresholds = []
        elif classification_type == 'multilabel':
            optimal_T = []
            optimal_thresholds = []
            for i in range(n_classes):
                optimal_T.append(trial.params[f'T_{i}'])
                optimal_thresholds.append(trial.params[f'conf_threshold_{i}'])
        return optimal_T, optimal_thresholds
    
   
def evaluate_predictions(preds, labels, classification_type, class_names =None):
    '''
    Evaluate the predictions of a model using the probabilities and the true labels.
    Parameters:
    preds: List/str, the predicted classes, must be one-hot encoded or the path to the saved predictions as a csv file
    labels: List/str, the true labels, must be one-hot encoded or the path to the saved labels as a csv file
    classification_type: str, type of classification problem; should be one of the following: 'multiclass', 'multilabel'
    class_names: list, list of class names to display in the classification report
    
    Returns:
    report: dict, the classification report
    '''
    
    if isinstance(preds, str):
        preds = pd.read_csv(preds)
        preds = preds.values
    if isinstance(labels, str):
        labels = pd.read_csv(labels)
        labels = labels.values
        
    if class_names == None:
        #read number of classes from the shape of the labels
        labels= np.array(labels)
        n_classes = labels.shape[1]
        class_names = [f'class{i}' for i in range(n_classes)]
        

    report = classification_report(labels, preds, output_dict=True, zero_division=0, target_names=class_names)
    
    y_true = np.array(labels)
    y_pred = np.array(preds)
    if classification_type == 'multiclass':
        #for multiclass, we calculate accuracy for the whole dataset, there is no class accuracy
        accuracy = np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
        #make all class accuracies the accuracy
        class_accuracies = [accuracy for i in range(y_true.shape[1])]
    elif classification_type == 'multilabel':
        # For multilabel, we calculate accuracy for each label
        n_classes = y_true.shape[1]
        class_accuracies = []
        for i in range(n_classes):
            true_positives = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            true_negatives = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
            total_samples = y_true.shape[0]
            class_acc = (true_positives + true_negatives) / total_samples
            class_accuracies.append(class_acc)
    # Calculate micro accuracy
    micro_accuracy = np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))
    # Calculate macro accuracy
    macro_accuracy = np.mean(class_accuracies)
    # Calculate weighted accuracy
    sample_weight = np.sum(y_true, axis=0) / len(y_true)
    weighted_accuracy = np.sum(class_accuracies * sample_weight) / np.sum(sample_weight)
    
    for i, class_name in enumerate(class_names):
        class_accuracy = class_accuracies[i]
        report[class_name]['accuracy'] = class_accuracy
    report['micro avg']['accuracy'] = micro_accuracy
    report['macro avg']['accuracy'] = macro_accuracy
    report['weighted avg']['accuracy'] = weighted_accuracy
    # reorder keys so accuracy is the first key
    for key in report.keys():
        metrics = report[key]
        if 'accuracy' in metrics:
            reordered_metrics = {'accuracy': metrics['accuracy']}
            for metric, value in metrics.items():
                if metric != 'accuracy':
                    reordered_metrics[metric] = value
            report[key] = reordered_metrics
    report.pop('samples avg')
    
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    return report

def create_classwise_bar_plot(report, savedir=None):
    #create a bar plot of the classwise metrics
    precisions = [report[cls]['precision'] for cls in report.keys()]
    recalls = [report[cls]['recall'] for cls in report.keys()]
    f1_scores = [report[cls]['f1-score'] for cls in report.keys()]
    accuracies = [report[cls]['accuracy'] for cls in report.keys()]
    ind = np.arange(len(report.keys()))
    width = 0.15
    fig, ax = plt.subplots(figsize=(7, 5))#first position is width of the bars second is height
    deep_colors = sns.color_palette('deep')
    accuracy_bars = ax.bar(ind - width, accuracies, width, label='Accuracy',color = deep_colors[0])
    precision_bars = ax.bar(ind , precisions, width, label='Precision',color = deep_colors[1])
    recall_bars = ax.bar(ind + width, recalls, width, label='Recall',color = deep_colors[2])
    f1score_bars = ax.bar(ind + 2*width, f1_scores, width, label='F1-score',color=deep_colors[3])
    # Adding some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Class Wise Evaluation Metrics')
    ax.set_xticks(ind)
    ax.set_xticklabels(report.keys())
    #make legend at botom right
    ax.legend(bbox_to_anchor=(1.01, 0.0), loc='lower left', borderaxespad=0)
    ax.set_ylim(0.0, 1)
    ax.set_yticks(np.arange(0.0, 1.05, 0.1))
    ax.set_yticks(np.arange(0.0, 1.05, 0.05), minor=True)
    #make grid of only horizontal lines with major and minor lines
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.75)
    ax.yaxis.grid(True, linestyle=':', which='minor', color='grey', alpha=.5)
    plt.xticks(rotation=90)
    plt.tight_layout()
    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(f'{savedir}/classwise_metrics_barplot.png')
    plt.show()


    
def visualize_predictions(model, data, img_size, classification_type, class_names=None, savedir=None):
    if isinstance(data, list):
        data = np.asarray(data)
    if isinstance(data, np.ndarray):
        #add batch dimension if not present
        if data.ndim == 3:
            data = np.expand_dims(data, axis=0)
        print('got {} images for prediction'.format(data.shape[0]))
    elif isinstance(data, str):
        assert img_size, 'img_size should be provided when using image paths'
        if os.path.isdir(data):
            img_files = [os.path.join(data,img) for img in os.listdir(data) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print('found',len(img_files),'images in the folder')
            img_arrays = [process_image(img) for img in img_files]
            img_array = np.vstack(img_arrays)
            #print('shape:',img_array.shape)
            
            
        elif os.path.isfile(data):
            img_array = process_image(data)
            probabilities = model.predict(img_array)
    else:
        raise ValueError('data should be a tf.data.Dataset, numpy array, image path or folder path')
    
    
def visualize_prediction(model, image):
        print('img shape: ',np.shape(image))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        print('img_array shape: ',img_array.shape)
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=-1)
        print('img_array shape: ',img_array.shape)
        #print('img min max: ',np.min(img_array),np.max(img_array))
        #covert to tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        replace2linear = ReplaceToLinear()
        # Create Saliency object.
        saliency = Saliency(x_model,
                            model_modifier=replace2linear,
                            clone=True)
        score = CategoricalScore([1])
        # Generate saliency map
        saliency_map = saliency(score, img_tensor, smooth_samples=100,smooth_noise=0.15)[0]
        print('saliency map shape: ',saliency_map.shape)
        print('min max: ',np.min(saliency_map),np.max(saliency_map))
        saliency_map = saliency_map * 255
        saliency_map = saliency_map.astype(np.uint8)
        print('min max: ',np.min(saliency_map),np.max(saliency_map))
        return {'saliency': saliency_map.tolist(), 'input_image': image.tolist()}