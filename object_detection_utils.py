import tensorflow as tf
import keras_cv
import keras
from keras_cv.models import object_detection as detection

def build_img_dataset_pipeline(img_path, label_path, n_classes, img_size=None, batch_size=16, augment_strength=0.1, 
                              val_size=0.2, test_size=0.1, shuffle=True, img_type='png', label_format='csv', 
                              seed=42, verbose=1, return_filenames=False):
    '''
    This function builds the training pipeline for object detection data. It reads the images from the image_path and labels from the label_path.
    It then performs the following operations:
    1. Reads the images and labels
    2. Resizes the images to img_size
    3. Augments the images and their corresponding bounding boxes
    4. Splits the data into training, validation and test sets
    5. Creates a tf.data.Dataset object for each set
    6. Batches the data
    7. Shuffles the data
    8. Returns the training, validation and test datasets
    
    Parameters:
    img_path: str, path to the folder that contains the images; images should be in .png or .jpeg format
    label_path: str, path to the file or directory that contains the labels
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
    label_format: str (optional, default: 'csv'), format of the labels; should be one of:
                  - 'csv': Custom CSV format with columns [image_name, class_id, x_min, y_min, x_max, y_max]
                  - 'pascal_voc': XML files in PASCAL VOC format, label_path should be a directory containing XML files
                  - 'yolo': TXT files in YOLO format, label_path should be a directory containing TXT files
                  - 'coco': JSON file in COCO format, label_path should be a JSON file
    seed: int (optional, default: 42), seed for shuffling the data, same seed will give same shuffling order
    verbose: int (optional, default:1), whether to show helpful messages
    return_filenames: bool (optional, default: False), whether to return the filenames of the images used in training, validation and test datasets
    
    Returns:
    train_ds: tf.data.Dataset, training dataset
    val_ds: tf.data.Dataset, validation dataset
    test_ds: tf.data.Dataset, test dataset
    if return_filenames==True returns additionally:
    train_files: list, list of filenames used in the training dataset
    val_files: list, list of filenames used in the validation dataset
    test_files: list, list of filenames used in the test dataset
    '''
    import os
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    import matplotlib.pyplot as plt
    import PIL
    import math
    from tqdm import tqdm
    from collections import defaultdict
    import json
    import xml.etree.ElementTree as ET
    import glob
    
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
    assert label_format in ['csv', 'pascal_voc', 'yolo', 'coco'], 'label_format should be one of: "csv", "pascal_voc", "yolo", "coco"'
    
    print(f'building training pipeline for object detection with {label_format} format...')
    
    # Function to parse different label formats and return a standardized format
    def parse_labels():
        img_files = []
        bbox_data = []
        class_names = {}
        
        # Custom CSV format
        if label_format == 'csv':
            labels_df = pd.read_csv(label_path)
            expected_columns = ['image_name', 'class_id', 'x_min', 'y_min', 'x_max', 'y_max']
            assert all(col in labels_df.columns for col in expected_columns), f'label file should contain columns: {expected_columns}'
            
            # Optional class name column
            if 'class_name' in labels_df.columns:
                for idx, name in zip(labels_df['class_id'].unique(), labels_df['class_name'].unique()):
                    class_names[idx] = name
            
            # Group by image_name to get all objects for each image
            grouped_labels = labels_df.groupby('image_name')
            
            for img_name, group in grouped_labels:
                img_files.append(os.path.join(img_path, img_name))
                
                # Extract bounding boxes and classes for this image
                boxes = group[['y_min', 'x_min', 'y_max', 'x_max']].values.astype(np.float32)
                classes = group['class_id'].values.astype(np.int32)
                
                bbox_data.append((boxes, classes))
        
        # PASCAL VOC format
        elif label_format == 'pascal_voc':
            xml_files = glob.glob(os.path.join(label_path, "*.xml"))
            
            for xml_file in xml_files:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Get filename
                filename = root.find("filename").text
                img_files.append(os.path.join(img_path, filename))
                
                # Get size
                size = root.find("size")
                width = int(size.find("width").text)
                height = int(size.find("height").text)
                
                boxes = []
                classes = []
                
                # Get objects
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    
                    # Assign class ID if not already in class_names
                    class_id = next((k for k, v in class_names.items() if v == class_name), None)
                    if class_id is None:
                        class_id = len(class_names)
                        class_names[class_id] = class_name
                    
                    bndbox = obj.find("bndbox")
                    xmin = float(bndbox.find("xmin").text) / width
                    ymin = float(bndbox.find("ymin").text) / height
                    xmax = float(bndbox.find("xmax").text) / width
                    ymax = float(bndbox.find("ymax").text) / height
                    
                    # PASCAL VOC format: [xmin, ymin, xmax, ymax] -> convert to [ymin, xmin, ymax, xmax]
                    boxes.append([ymin, xmin, ymax, xmax])
                    classes.append(class_id)
                
                if boxes:  # Only add images with annotations
                    bbox_data.append((np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)))
        
        # YOLO format
        elif label_format == 'yolo':
            # YOLO format: Each image has a corresponding .txt file with the same name
            # Each line in the .txt file is: class_id x_center y_center width height (all normalized [0,1])
            label_files = glob.glob(os.path.join(label_path, "*.txt"))
            
            for label_file in label_files:
                basename = os.path.basename(label_file)
                img_name = os.path.splitext(basename)[0] + f'.{img_type}'
                img_path_full = os.path.join(img_path, img_name)
                
                if os.path.exists(img_path_full):
                    img_files.append(img_path_full)
                    
                    boxes = []
                    classes = []
                    
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                
                                # Convert YOLO format to [ymin, xmin, ymax, xmax]
                                xmin = x_center - width/2
                                ymin = y_center - height/2
                                xmax = x_center + width/2
                                ymax = y_center + height/2
                                
                                # Clip values to ensure they're in [0,1]
                                xmin = max(0, min(1, xmin))
                                ymin = max(0, min(1, ymin))
                                xmax = max(0, min(1, xmax))
                                ymax = max(0, min(1, ymax))
                                
                                boxes.append([ymin, xmin, ymax, xmax])
                                classes.append(class_id)
                    
                    if boxes:  # Only add images with annotations
                        bbox_data.append((np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)))
        
        # COCO format
        elif label_format == 'coco':
            with open(label_path, 'r') as f:
                coco_data = json.load(f)
            
            # Create mappings
            img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
            img_id_to_size = {img['id']: (img['height'], img['width']) for img in coco_data['images']}
            
            # Get class names
            for category in coco_data['categories']:
                class_names[category['id']] = category['name']
            
            # Group annotations by image_id
            annotations_by_image = defaultdict(list)
            for ann in coco_data['annotations']:
                annotations_by_image[ann['image_id']].append(ann)
            
            # Process each image
            for img_id, annotations in annotations_by_image.items():
                filename = img_id_to_filename[img_id]
                img_files.append(os.path.join(img_path, filename))
                
                height, width = img_id_to_size[img_id]
                
                boxes = []
                classes = []
                
                for ann in annotations:
                    # COCO bbox format is [x, y, width, height]
                    x, y, w, h = ann['bbox']
                    
                    # Convert to normalized coordinates [ymin, xmin, ymax, xmax]
                    xmin = x / width
                    ymin = y / height
                    xmax = (x + w) / width
                    ymax = (y + h) / height
                    
                    boxes.append([ymin, xmin, ymax, xmax])
                    classes.append(ann['category_id'])
                
                if boxes:  # Only add images with annotations
                    bbox_data.append((np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)))
        
        assert len(img_files) > 0, f"No valid images found with {label_format} annotations"
        print(f'Found {len(img_files)} images with object annotations')
        
        return img_files, bbox_data, class_names
    
    # Parse labels based on the specified format
    img_files, bbox_data, class_names = parse_labels()
    
    # If class_names is empty, create generic names
    if not class_names and n_classes > 0:
        class_names = {i: f"Class {i}" for i in range(n_classes)}
    elif class_names and n_classes == 0:
        n_classes = len(class_names)
    
    # Define a function to read and preprocess the images
    def parse_object_detection_data(image_path, bbox_data):
        # Read the image
        image = tf.io.read_file(image_path)
        if img_type == 'png':
            image = tf.image.decode_png(image, channels=img_size[2])
        elif img_type == 'jpeg':
            image = tf.image.decode_jpeg(image, channels=img_size[2])
        
        # Get original dimensions
        original_height = tf.cast(tf.shape(image)[0], tf.float32)
        original_width = tf.cast(tf.shape(image)[1], tf.float32)
        
        # Resize the image
        image = tf.image.resize(image, (img_size[0], img_size[1]), method='area')
        
        # Unpack the bbox_data
        boxes, classes = bbox_data
        
        # Ensure boxes are normalized to [0, 1]
        boxes = tf.clip_by_value(boxes, 0.0, 1.0)
        
        return image, (boxes, classes)
    
    # Function to apply augmentation while maintaining bbox coordinates
    def augment_with_bboxes(image, bbox_data):
        boxes, classes = bbox_data
        
        # Apply custom augmentations that handle both image and bounding boxes
        # Random flip
        if augment_strength[0] > 0:
            # Horizontal flip
            if augment_strength[0] in [1, 2]:
                flip_lr = tf.random.uniform([], 0, 1) > 0.5
                if flip_lr:
                    image = tf.image.flip_left_right(image)
                    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)
                    flipped_x_min = 1.0 - x_max
                    flipped_x_max = 1.0 - x_min
                    boxes = tf.concat([y_min, flipped_x_min, y_max, flipped_x_max], axis=1)
            
            # Vertical flip
            if augment_strength[0] in [1, 3]:
                flip_ud = tf.random.uniform([], 0, 1) > 0.5
                if flip_ud:
                    image = tf.image.flip_up_down(image)
                    y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=1)
                    flipped_y_min = 1.0 - y_max
                    flipped_y_max = 1.0 - y_min
                    boxes = tf.concat([flipped_y_min, x_min, flipped_y_max, x_max], axis=1)
        
        # Apply other augmentations that don't affect bounding box coordinates
        # Random contrast
        if augment_strength[3] > 0:
            image = tf.image.random_contrast(image, 1-augment_strength[3], 1+augment_strength[3])
        
        # Random brightness
        if augment_strength[4] > 0:
            image = tf.image.random_brightness(image, augment_strength[4])
        
        # Gaussian noise
        if augment_strength[5] > 0:
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=augment_strength[5], dtype=tf.float32)
            image = image + noise
        
        # Ensure image values are in valid range
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, tf.uint8)
        
        return image, (boxes, classes)
    
    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices((img_files, bbox_data))
    
    if shuffle:
        dataset = dataset.shuffle(len(img_files), reshuffle_each_iteration=False, seed=seed)
    
    # Handle split sizes
    if val_size == 1.0:
        test_size = 0.0
    if test_size == 1.0:
        val_size = 0.0
        
    # Split dataset into train, validation and test sets
    test_nr = int(test_size * len(img_files))
    val_nr = int(val_size * len(img_files))
    train_nr = len(img_files) - val_nr - test_nr
    
    print('nr of training samples:', train_nr)
    print('nr of validation samples:', val_nr)
    print('nr of test samples:', test_nr)
    
    if train_nr == 0:
        train_ds = None
    else:
        train_ds = dataset.take(train_nr)
    
    remaining_ds = dataset.skip(train_nr)
    
    if test_size:
        val_ds = remaining_ds.take(val_nr) if val_nr > 0 else None
        test_ds = remaining_ds.skip(val_nr)
    else:
        val_ds = remaining_ds if val_nr > 0 else None
        test_ds = None
    
    # Prepare and visualize dataset information
    if verbose:
        # Collect information for visualization
        class_counts = defaultdict(int)
        sample_images = {}
        
        # Count instances of each class
        for _, (boxes, classes) in bbox_data:
            for class_id in classes:
                class_counts[class_id] += 1
                if class_id not in sample_images and len(sample_images) < n_classes:
                    sample_idx = next((i for i, (_, (_, cls)) in enumerate(bbox_data) if class_id in cls), None)
                    if sample_idx is not None:
                        sample_images[class_id] = img_files[sample_idx]
        
        print("Class distribution:")
        for class_id, count in sorted(class_counts.items()):
            class_label = class_names.get(class_id, f"Class {class_id}")
            print(f"{class_label} (ID {class_id}): {count} objects")
        
        # Plot sample images with bounding boxes if available
        if sample_images:
            plt.figure(figsize=(20, 10))
            for i, (class_id, img_file) in enumerate(sample_images.items()):
                if i >= n_classes:
                    break
                    
                plt.subplot(2, math.ceil(min(len(sample_images), n_classes)/2), i+1)
                
                # Find the original bounding boxes for this image
                img_idx = img_files.index(img_file)
                boxes, classes = bbox_data[img_idx]
                
                # Load and display the image
                img = PIL.Image.open(img_file)
                img = img.resize((img_size[1], img_size[0]))
                img_array = np.array(img)
                
                plt.imshow(img_array)
                
                # Draw bounding boxes
                img_height, img_width = img_array.shape[:2]
                for box_idx, cls in enumerate(classes):
                    if box_idx < len(boxes):
                        # Convert normalized coordinates to pixel coordinates
                        y_min, x_min, y_max, x_max = boxes[box_idx]
                        
                        # Convert normalized coordinates [0,1] to pixel values
                        y_min *= img_height
                        x_min *= img_width
                        y_max *= img_height
                        x_max *= img_width
                        
                        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                          fill=False, edgecolor='red', linewidth=2)
                        plt.gca().add_patch(rect)
                        class_label = class_names.get(cls, f"Class {cls}")
                        plt.text(x_min, y_min - 5, f"{class_label}", 
                               color='white', backgroundcolor='red', fontsize=8)
                
                plt.axis('off')
                plt.title(f'{class_names.get(class_id, f"Class {class_id}")} example')
            
            plt.suptitle('Sample images with bounding boxes')
            plt.tight_layout()
            plt.show()
            
            # If augmentation is enabled, show augmented examples
            if train_ds and augment_strength[0] > 0:
                # Create a function to show augmented images
                def visualize_augmentation(image_path, bbox_data):
                    # Read and process image
                    image, (boxes, classes) = parse_object_detection_data(image_path, bbox_data)
                    # Apply augmentation several times
                    augmented_samples = []
                    
                    for _ in range(4):
                        aug_image, (aug_boxes, aug_classes) = augment_with_bboxes(tf.identity(image), (tf.identity(boxes), classes))
                        augmented_samples.append((aug_image, aug_boxes, aug_classes))
                    
                    return image, boxes, classes, augmented_samples
                
                # Select a sample image
                sample_img_path = next(iter(sample_images.values()))
                sample_idx = img_files.index(sample_img_path)
                
                original_image, original_boxes, original_classes, augmented_samples = visualize_augmentation(
                    sample_img_path, bbox_data[sample_idx])
                
                # Plot original and augmented images
                plt.figure(figsize=(20, 10))
                
                # Original image
                plt.subplot(1, 5, 1)
                plt.imshow(original_image.numpy().astype(np.uint8))
                
                # Draw original boxes
                img_height, img_width = original_image.shape[:2]
                for i, cls in enumerate(original_classes):
                    y_min, x_min, y_max, x_max = original_boxes[i].numpy()
                    y_min *= img_height
                    x_min *= img_width
                    y_max *= img_height
                    x_max *= img_width
                    
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      fill=False, edgecolor='red', linewidth=2)
                    plt.gca().add_patch(rect)
                
                plt.title("Original")
                plt.axis('off')
                
                # Augmented images
                for i, (aug_image, aug_boxes, aug_classes) in enumerate(augmented_samples):
                    plt.subplot(1, 5, i+2)
                    plt.imshow(aug_image.numpy().astype(np.uint8))
                    
                    # Draw augmented boxes
                    img_height, img_width = aug_image.shape[:2]
                    for j, cls in enumerate(aug_classes):
                        y_min, x_min, y_max, x_max = aug_boxes[j].numpy()
                        y_min *= img_height
                        x_min *= img_width
                        y_max *= img_height
                        x_max *= img_width
                        
                        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                          fill=False, edgecolor='red', linewidth=2)
                        plt.gca().add_patch(rect)
                    
                    plt.title(f"Augmented {i+1}")
                    plt.axis('off')
                
                plt.suptitle('Original vs Augmented Images with Bounding Boxes')
                plt.tight_layout()
                plt.show()
    
    # Process datasets
    if train_ds:
        train_ds = train_ds.map(parse_object_detection_data, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(augment_with_bboxes, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    if val_ds:
        val_ds = val_ds.map(parse_object_detection_data, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    if test_ds:
        test_ds = test_ds.map(parse_object_detection_data, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size)
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Prepare filename lists if requested
    if return_filenames:
        print('extracting filenames...')
        train_files, val_files, test_files = [], [], []
        
        if train_ds:
            # We need to use the unbatched version to extract filenames
            for img_path, _ in dataset.take(train_nr).as_numpy_iterator():
                train_files.append(os.path.basename(img_path))
        
        if val_ds:
            for img_path, _ in dataset.skip(train_nr).take(val_nr).as_numpy_iterator():
                val_files.append(os.path.basename(img_path))
        else:
            val_files = None
            
        if test_ds:
            for img_path, _ in dataset.skip(train_nr + val_nr).take(test_nr).as_numpy_iterator():
                test_files.append(os.path.basename(img_path))
        else:
            test_files = None
        
        return train_ds, val_ds, test_ds, train_files, val_files, test_files
    else:
        return train_ds, val_ds, test_ds
    
from tensorflow.keras.models import Sequential
import tensorflow as tf
import os
import json
import csv
import xml.etree.ElementTree as ET
from tqdm import tqdm
import keras_cv
import keras
import math
from collections import defaultdict
from PIL import Image
import matplotlib.patches as patches

    

def build_image_detection_pipeline(img_path=None, labels_path=None, annotation_format="yolo", 
                                  img_size=None, batch_size=32, augment_strength={'rotation':0.1,
                                                                                       'shear':0.1,
                                                                                       'hue':0.1,
                                                                                       'saturation':(0.4,0.6),
                                                                                       'contrast':0.1,
                                                                                       'brightness':0.1,
                                                                                       'blur':0.1,
                                                                                       'cutout':0.1}):
    """
    Build an object detection pipeline supporting multiple annotation formats
    
    Parameters:
    - img_path: Path to images
    - labels_path: Path to annotations
    - annotation_format: One of "yolo", "pascal_voc", "coco", "csv"
    - img_size: Image dimensions (height, width)
    - batch_size: Batch size for training
    
    Returns:
    - train_ds: Training dataset
    - val_ds: Validation dataset
    - class_mapping: Dictionary mapping class IDs to class names
    """
    if img_size == None:
        #infer size from first image
        img = Image.open(os.path.join(img_path,os.listdir(img_path)[0]))
        img_size = img.size
 
    # YOLO format parser (original)
    def parse_yolo_labels(label_path):
        label = open(label_path, 'r').read().split('\n')
        
        class_ids = []
        bbs = []
        for line in label:
            if line == '':
                continue
            #first string before space is the class, after that is the bounding box
            class_id = int(line.split(' ')[0])
            # Check if class_id is a number or string
            #read bounding box
            bb = line.split(' ')[1:]
            bb = [float(i) for i in bb]
            #clip values to 0 and 1
            bb = [max(0,i) for i in bb]
            bb = [min(1,i) for i in bb]
            
            # YOLO format is x_center, y_center, width, height
            # Convert to x1, y1, x2, y2 format
            x_center, y_center, width, height = bb
            x1 = (x_center - width/2) * img_size[0]
            y1 = (y_center - height/2) * img_size[1]
            x2 = (x_center + width/2) * img_size[0]
            y2 = (y_center + height/2) * img_size[1]
            
            class_ids.append(class_id)
            bbs.append([x1, y1, x2, y2])
            
        return class_ids, bbs
    
    # Pascal VOC format parser
    def parse_pascal_voc_labels(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        class_ids = []
        bbs = []
        
        for obj in root.findall('object'):
            class_id = obj.find('name').text
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            
            class_ids.append(class_id)
            bbs.append([x1, y1, x2, y2])
            
        return class_ids, bbs
    
    # COCO format parser
    def parse_coco_labels(img_file, coco_json_path):
        # Extract image filename
        img_filename = os.path.basename(img_file)
        
        # Load COCO JSON
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Find image ID
        img_id = None
        for img in coco_data['images']:
            if img['file_name'] == img_filename:
                img_id = img['id']
                img_width = img['width']
                img_height = img['height']
                break
        
        if img_id is None:
            return [], []
        
        class_ids = []
        bbs = []
        
        # Find annotations for this image
        for ann in coco_data['annotations']:
            if ann['image_id'] == img_id:
                # COCO uses category_id
                category_id = ann['category_id']
                
                # Map COCO category to our class list
                # We need to find the category name first
                category_name = None
                for cat in coco_data['categories']:
                    if cat['id'] == category_id:
                        category_name = cat['name']
                        break
                
                class_id = category_name
    
                # COCO format: [x, y, width, height]
                x, y, width, height = ann['bbox']
                
                # Convert to x1, y1, x2, y2
                x1 = x * img_size[0] / img_width
                y1 = y * img_size[1] / img_height
                x2 = (x + width) * img_size[0] / img_width
                y2 = (y + height) * img_size[1] / img_height
                
                class_ids.append(class_id)
                bbs.append([x1, y1, x2, y2])
        
        return class_ids, bbs
    
    # CSV format parser
    def parse_csv_labels(csv_file, img_file):
        img_filename = os.path.basename(img_file)
        
        class_ids = []
        bbs = []
        
        with open(csv_file, 'r') as f:
            csv_reader = csv.reader(f)
            # Skip header if exists
            header = next(csv_reader, None)
            
            for row in csv_reader:
                # CSV format can vary, but typically:
                # filename, class, xmin, ymin, xmax, ymax
                if len(row) >= 6 and row[0] == img_filename:
                    class_id = row[1]
                    
                    # Parse coordinates
                    xmin, ymin, xmax, ymax = map(float, row[2:6])
                    
                    # Scale to our image size
                    x1 = xmin * img_size[0]
                    y1 = ymin * img_size[1]
                    x2 = xmax * img_size[0]
                    y2 = ymax * img_size[1]
                    
                    class_ids.append(class_id)
                    bbs.append([x1, y1, x2, y2])
        
        return class_ids, bbs



    # Validate required paths
    if img_path is None or labels_path is None:
        raise ValueError("Both img_path and labels_path must be provided")
    
    # Get image files
    files = sorted(os.listdir(img_path))
    files = [os.path.join(img_path, file) for file in files]
    
    class_ids = []
    bbs = []
    
    # Parse annotations based on format
    if annotation_format.lower() == "yolo":
        label_files = sorted(os.listdir(labels_path))
        label_files = [os.path.join(labels_path, label) for label in label_files]
        
        for label in tqdm(label_files):
            class_id, bb = parse_yolo_labels(label)
            class_ids.append(class_id)
            bbs.append(bb)
            
    elif annotation_format.lower() == "pascal_voc":
        xml_files = sorted(os.listdir(labels_path))
        xml_files = [os.path.join(labels_path, xml) for xml in xml_files]
        
        for xml in tqdm(xml_files):
            class_id, bb = parse_pascal_voc_labels(xml)
            class_ids.append(class_id)
            bbs.append(bb)
            
    elif annotation_format.lower() == "coco":
        # For COCO, labels_path should be the path to the JSON file
        for img_file in tqdm(files):
            class_id, bb = parse_coco_labels(img_file, labels_path)
            class_ids.append(class_id)
            bbs.append(bb)
            
    elif annotation_format.lower() == "csv":
        # For CSV, labels_path should be the path to the CSV file
        for img_file in tqdm(files):
            class_id, bb = parse_csv_labels(labels_path, img_file)
            class_ids.append(class_id)
            bbs.append(bb)
            
    else:
        raise ValueError(f"Unsupported annotation format: {annotation_format}. " +
                         "Supported formats: yolo, pascal_voc, coco, csv")
    
    
    def parse_images(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image,channels=3) 
        #image = tf.image.resize(image, img_size)
        return image
    
    def load_dataset(img_file, id, bb):
        image = parse_images(img_file)
        #bb to normal tensor
        bounding_boxes = {"boxes": tf.cast(bb, dtype=tf.float32), "classes": tf.cast(id, dtype=tf.float32)}
        bounding_boxes = keras_cv.bounding_box.to_dense(bounding_boxes)
        return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}
    
    #create class mapping from class_ids string -> int
    class_mapping = {}
    for class_id in class_ids:
        for id in class_id:
            if id not in class_mapping:
                class_mapping[id] = len(class_mapping)
    print('class_mapping:',class_mapping)
    class_ids = [[class_mapping[id] for id in ids] for ids in class_ids]
    class_ids = tf.ragged.constant(class_ids)
    bbs = tf.ragged.constant(bbs)
    print('bbs:',bbs[:5])
    print('class_ids:',class_ids[:5])
    files = tf.ragged.constant(files)
    dataset = tf.data.Dataset.from_tensor_slices((files, class_ids, bbs))
    dataset = dataset.shuffle(len(files))

    train_size = int(0.7 * len(files))
    val_size = int(0.3 * len(files))
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)
    
    augmentor = keras_cv.layers.Augmenter(
        [
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
            #keras_cv.layers.RandomRotation( fill_mode="constant", bounding_box_format="xyxy", factor=augment_strength['rotation']),
            keras_cv.layers.RandomShear(bounding_box_format="xyxy",x_factor=augment_strength['shear'],y_factor=augment_strength['shear']),
            keras_cv.layers.RandomHue(value_range=(0,255),factor=augment_strength['hue']),
            keras_cv.layers.RandomSaturation(factor=(augment_strength['saturation'])),
            keras_cv.layers.RandomContrast(value_range=(0,255),factor=augment_strength['contrast']),
            keras_cv.layers.RandomBrightness(value_range=(0,255),factor=augment_strength['brightness']),
            keras_cv.layers.RandomGaussianBlur(kernel_size=3, factor=augment_strength['blur']),
            keras_cv.layers.RandomCutout(height_factor=augment_strength['cutout'],width_factor=augment_strength['cutout'],fill_mode="gaussian_noise"),
            keras_cv.layers.JitteredResize(target_size=(img_size[0],img_size[1]), scale_factor=(0.7, 1.3), bounding_box_format="xyxy"),
            #keras_cv.layers.Rescaling(scale=1./255)
        ]
    )
    resizing = keras.Sequential(layers=[keras_cv.layers.JitteredResize(target_size=(img_size[0],img_size[1]),scale_factor=(0.75, 1.3),bounding_box_format="xyxy"),
                                        #keras_cv.layers.Rescaling(scale=1./255)
                                        ]
                                )

    
    train_ds = train_ds.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.ragged_batch(batch_size, drop_remainder=True)
    train_ds = train_ds.map(augmentor, num_parallel_calls=tf.data.AUTOTUNE)
    
    
    val_ds = val_ds.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.ragged_batch(batch_size, drop_remainder=True)
    val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    def visualize_dataset(inputs, rows, cols):
        inputs = next(iter(inputs.take(1)))
        images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
        keras_cv.visualization.plot_bounding_box_gallery(
            images,
            value_range=(0,255),
            rows=rows,
            cols=cols,
            y_true=bounding_boxes,
            scale=3,
            font_scale=1,
            bounding_box_format='xyxy',
            class_mapping=class_mapping,
        )
    
    visualize_dataset(train_ds, 5, 4)
    visualize_dataset(val_ds, 5, 4)
    
    def dict_to_tuple(inputs):
        return inputs["images"], keras_cv.bounding_box.to_dense(
            inputs["bounding_boxes"], max_boxes=32
        )
    
    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    
    return train_ds, val_ds
