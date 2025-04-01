import tensorflow as tf
import keras_cv
import keras
print("tf.__version__:", tf.__version__)
print("keras.__version__:", keras.__version__)
print("keras_cv.__version__:", keras_cv.__version__)
import keras_hub
import tensorflow_hub as hub
from datetime import datetime
from PIL import Image
import os
import json
import csv
import xml.etree.ElementTree as ET
from tqdm import tqdm
import keras_cv

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
            keras_cv.layers.Resizing(height=img_size[0], width=img_size[1], pad_to_aspect_ratio=True, bounding_box_format="xyxy",),
            keras_cv.layers.JitteredResize(target_size=(img_size[0],img_size[1]), scale_factor=(0.7, 1.3), bounding_box_format="xyxy"),
            #keras_cv.layers.Rescaling(scale=1./255)
        ]
    )
    resizing = keras.Sequential(layers=[keras_cv.layers.Resizing(height=img_size[0], width=img_size[1], pad_to_aspect_ratio=True, bounding_box_format="xyxy"),
                                        keras_cv.layers.JitteredResize(target_size=(img_size[0],img_size[1]),scale_factor=(0.75, 1.3),bounding_box_format="xyxy"),
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
    
    rows = min(batch_size // 5, 4)
    
    visualize_dataset(train_ds, 5, rows)
    visualize_dataset(val_ds, 5, rows)
    
    def dict_to_tuple(inputs):
        return inputs["images"], keras_cv.bounding_box.to_dense(
            inputs["bounding_boxes"], max_boxes=32
        )
    
    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    
    return train_ds, val_ds


    

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


def get_backbone(type, weights, model_size):
    hub_presets = {
        ('retinanet', 'coco'): "retinanet_resnet50_fpn_v2_coco",
        ('retinanet', 'voc'): "retinanet_resnet50_pascalvoc",
        ('faster_rcnn', 'coco'): "faster_rcnn_resnet50_fpn_coco",
        ('faster_rcnn', 'voc'): "faster_rcnn_resnet50_pascalvoc"
    }
    if (type, weights) in hub_presets:
        preset = hub_presets[(type, weights)]
    resnet_size_map = {
    's': '50',    # Small = ResNet-50
    'm': '101',   # Medium = ResNet-101
    'l': '152',   # Large = ResNet-152
    'xl': '152'   # Extra Large = ResNet-152 (fallback to largest available)
    }
    
    # Map model_size to EfficientNetV1 variant
    efficientnetv1_size_map = {
        's': '0',  # Small = B0
        'm': '1',  # Medium = B1
        'l': '2',  # Large = B2
        'xl': '3'  # Extra Large = B3
    }
    
    # Backbone initialization functions - using lambdas for lazy initialization
    backbone_factories = {
        # ResNet V2 family - using size parameter
        'resnetv2': {
            'imagenet': keras_cv.models.ResNet50V2Backbone.from_preset("resnet50_v2_imagenet"),
            None: {
                'xs': keras_cv.models.ResNet50V2Backbone.from_preset("resnet18_v2"),
                's': keras_cv.models.ResNet50V2Backbone.from_preset("resnet34_v2"),
                'm': keras_cv.models.ResNet50V2Backbone.from_preset("resnet50_v2"),
                'l': keras_cv.models.ResNet50V2Backbone.from_preset("resnet101_v2"),
            } 
        },
        # CSP DarkNet
        'cspdarknet': {
            'imagenet': {
                's': keras_cv.models.CSPDarkNetBackbone.from_preset("csp_darknet_tiny_imagenet"),
                'l': keras_cv.models.CSPDarkNetBackbone.from_preset("csp_darknet_l_imagenet")
                },
            None: {
                's': keras_cv.models.CSPDarkNetBackbone.from_preset("csp_darknet_tiny"),
                'm': keras_cv.models.CSPDarkNetBackbone.from_preset("csp_darknet_s"),
                'l': keras_cv.models.CSPDarkNetBackbone.from_preset("csp_darknet_m"),
                'xl': keras_cv.models.CSPDarkNetBackbone.from_preset("csp_darknet_l")
                }
        },
        
        'efficientnet': {
            None: {
                's': keras_cv.models.EfficientNetV2Backbone.from_preset(f"efficientnetv2_b0"),
                'm': keras_cv.models.EfficientNetV2Backbone.from_preset(f"efficientnetv2_b1"),
                'l': keras_cv.models.EfficientNetV2Backbone.from_preset(f"efficientnetv2_b2"),
                'xl': keras_cv.models.EfficientNetV2Backbone.from_preset(f"efficientnetv2_b3")
                },
            'imagenet': {
                's': keras_cv.models.EfficientNetV2Backbone.from_preset(f"efficientnetv2_b0_imagenet"),
                'm': keras_cv.models.EfficientNetV2Backbone.from_preset(f"efficientnetv2_b1_imagenet"),
                'l': keras_cv.models.EfficientNetV2Backbone.from_preset(f"efficientnetv2_b2_imagenet"),
                },
        },
        'efficientnetlite': {
            None: {
                's': lambda: keras_cv.models.EfficientNetLiteB0Backbone(include_rescaling=False),  # Direct instantiation
                'm': lambda: keras_cv.models.EfficientNetLiteB1Backbone(include_rescaling=False),  # Direct instantiation
                'l': lambda: keras_cv.models.EfficientNetLiteB2Backbone(include_rescaling=False),  # Direct instantiation
                'xl': lambda: keras_cv.models.EfficientNetLiteB3Backbone(include_rescaling=False) # Direct instantiation
                # Note: EfficientNetLiteB4Backbone might exist, adjust 'xl' if needed.
            }
            # Add 'imagenet' key here if you plan to support pretrained EfficientNetLite backbones later
        },
        
        # MobileNet
        'mobilenet': {
            None: {
                's': keras_cv.models.EfficientNetV2Backbone.from_preset(f"mobilenet_v3_small"),
                'l': keras_cv.models.EfficientNetV2Backbone.from_preset(f"mobilenet_v3_large"),
                },
            'imagenet': {
                's': keras_cv.models.EfficientNetV2Backbone.from_preset(f"mobilenet_v3_small_imagenet"),
                'l': keras_cv.models.EfficientNetV2Backbone.from_preset(f"mobilenet_v3_large_imagenet"),
            },
        },
        
        # YOLO v8 specific backbones
        'yolov8': {
            'coco':{
                's': keras_cv.models.YOLOV8Backbone.from_preset(f"yolo_v8_xs_backbone_coco"),
                'm': keras_cv.models.YOLOV8Backbone.from_preset(f"yolo_v8_s_backbone_coco"),
                'l': keras_cv.models.YOLOV8Backbone.from_preset(f"yolo_v8_m_backbone_coco"),
                },
            'voc': keras_cv.models.YOLOV8Backbone.from_preset(f"yolo_v8_m_pascalvoc"),
            None:{ 
                's': keras_cv.models.YOLOV8Backbone.from_preset(f"yolo_v8_xs_backbone"),
                'm': keras_cv.models.YOLOV8Backbone.from_preset(f"yolo_v8_s_backbone"),
                'l': keras_cv.models.YOLOV8Backbone.from_preset(f"yolo_v8_m_backbone"),
                }
        }
    }
    

def build_detection_model(n_classes, lr=0.001, model_name='ssd',weights='coco',model_size='s',backbone='standard'):
    '''
    This function builds a model for object detection using Keras CV.
    
    Parameters:
    input_shape: tuple, shape of the input images; should be in the format (height, width, channels)
    n_classes: int, number of object classes to detect (excluding background class)
    model_name: str (optional, default: 'ssd'), name of the detection model
                Options: 'ssd', 'retinanet', 'faster_rcnn', 'yolov8'
    
    Returns:
    model: keras.Model, the model for object detection
    '''
    
    assert model_name in ['ssd', 'retinanet', 'faster_rcnn', 'yolov8'], f"Model {model_name} not supported"
    assert weights in ['coco', 'imagenet','voc', None], f"Weights {weights} not supported"
    assert model_size in ['s', 'm', 'l', 'xl'], f"Model size {model_size} not supported"
    
    # if model_name == 'ssd':
    #     model = detection.SSD(input_shape, num_classes=n_classes)
    if model_name == 'retinanet':
        if backbone == 'standard':
            if weights == 'coco':
                model =  keras_hub.models.ImageObjectDetector.from_preset("retinanet_resnet50_fpn_v2_coco", num_classes=n_classes, bounding_box_format="xyxy")
            elif weights == 'voc':
                model =  keras_hub.models.ImageObjectDetector.from_preset("retinanet_resnet50_pascalvoc", num_classes=n_classes, bounding_box_format="xyxy")
            elif weights == 'imagenet':
                model = keras_cv.models.RetinaNet(num_classes=n_classes, bounding_box_format="xyxy",
                                                backbone=keras_cv.models.ResNet50Backbone.from_preset("resnet50_v2_imagenet"),)   
            elif weights == None:
                model = keras_cv.models.RetinaNet(num_classes=n_classes, bounding_box_format="xyxy",
                                                backbone=keras_cv.models.ResNet50V2Backbone(),)   
        else:
            backbone = get_backbone(backbone, weights, model_size)
            model = keras_cv.models.RetinaNet(num_classes=n_classes, bounding_box_format="xyxy", backbone=backbone)
            
    elif model_name == 'faster_rcnn':
        if backbone == 'standard':
            if weights == 'coco':
                model =  keras_hub.models.ImageObjectDetector.from_preset("faster_rcnn_resnet50_fpn_coco", num_classes=n_classes, bounding_box_format="xyxy")
            elif weights == 'voc':
                model =  keras_hub.models.ImageObjectDetector.from_preset("faster_rcnn_resnet50_pascalvoc", num_classes=n_classes, bounding_box_format="xyxy")
            elif weights == 'imagenet':
                model = keras_cv.models.FasterRCNN(num_classes=n_classes, bounding_box_format="xyxy",
                                                backbone=keras_cv.models.ResNet50Backbone.from_preset("resnet50_v2_imagenet"),)   
            elif weights == None:
                model = keras_cv.models.FasterRCNN(num_classes=n_classes, bounding_box_format="xyxy",
                                                backbone=keras_cv.models.ResNet50V2Backbone(),)   
        else:
            backbone = get_backbone(backbone, weights, model_size)
            model = keras_cv.models.FasterRCNN(num_classes=n_classes, bounding_box_format="xyxy", backbone=backbone)
        
    elif model_name == 'yolov8':
        if backbone == 'standard':
            if weights == 'coco':
                model = keras_cv.models.YOLOV8Detector(num_classes=n_classes, bounding_box_format="xyxy", backbone=keras_cv.models.YOLOV8Backbone.from_preset(f"yolo_v8_{model_size}_backbone_coco"),)
            if weights == 'voc':
                model = keras_cv.models.YOLOV8Detector(num_classes=n_classes, bounding_box_format="xyxy",  backbone=keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_m_pascalvoc",))
            #raise ValueError("VOC weights not supported for YOLOV8")
            if weights == 'imagenet':
                model = keras_cv.models.YOLOV8Detector(num_classes=n_classes, bounding_box_format="xyxy",  backbone=keras_cv.models.ResNet50Backbone.from_preset("resnet50_v2_imagenet"),)
            if weights == None:
                model = keras_cv.models.YOLOV8Detector(num_classes=n_classes, bounding_box_format="xyxy",  backbone=keras_cv.models.YOLOV8Backbone.from_preset(f"yolo_v8_{model_size}_backbone"),)
        elif backbone == 'resnet':
            if weights == 'coco':
                model = keras_cv.models.YOLOV8Detector(num_classes=n_classes, bounding_box_format="xyxy", backbone=keras_cv.models.ResNet50Backbone.from_preset("resnet50_v2_coco"),)
            if weights == 'voc':
                model = keras_cv.models.YOLOV8Detector(num_classes=n_classes, bounding_box_format="xyxy", backbone=keras_cv.models.ResNet50Backbone.from_preset("resnet50_v2_pascalvoc"),)
            if weights == 'imagenet':
                model = keras_cv.models.YOLOV8Detector(num_classes=n_classes, bounding_box_format="xyxy", backbone=keras_cv.models.ResNet50Backbone.from_preset("resnet50_v2_imagenet"),)
            if weights == None:
                model = keras_cv.models.YOLOV8Detector(num_classes=n_classes, bounding_box_format="xyxy", backbone=keras_cv.models.ResNet50Backbone(),)    
        else:
            backbone = get_backbone(backbone, weights, model_size)
            model = keras_cv.models.YOLOV8Detector(num_classes=n_classes, bounding_box_format="xyxy", backbone=backbone)
            
    
    return model    

def compile_model(model, optimizer='adam', lr=0.001,verbose=0):
    if optimizer == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)
    elif optimizer == 'lamb':
        optimizer = keras.optimizers.LAMB(learning_rate=lr)  
    
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, box_loss="ciou",classification_loss=loss)
    
    if verbose:
        model.summary()

    return model

def train_detection_model(model, train_ds, val_ds, epochs=10, verbose=1,folder='projects'):
    '''
    This function trains a model for object detection using Keras CV.
    
    Parameters:
    model: keras.Model, the model for object detection
    train_ds: tf.data.Dataset, the training dataset
    val_ds: tf.data.Dataset, the validation dataset
    epochs: int, number of epochs for training
    verbose: int, whether to print training information
    
    Returns:
    history: dict, history of training metrics
    '''
    save_path = r'../Data/trained_models'
    model_name = model.name
    if 'yolov8' in model_name:
        model_name = 'yolov8'
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    savedir = os.path.join(save_path, folder, datetime_str, model_name)
    print('savedir:',savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
        
    checkpoints = os.path.join(savedir, 'best_weights')
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints, exist_ok=True)
    # Train the model
    
    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoints+'/checkpoint.weights.h5'),save_best_only=True,save_weights_only=True,monitor='val_loss',mode='min',verbose=1))
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True))
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='min', min_delta=0.001, cooldown=0, min_lr=0))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=verbose,callbacks=callbacks)
    #load best weights
    model.load_weights(os.path.join(checkpoints+'/checkpoint.weights.h5'))
    # Save the model
    model.save(os.path.join(savedir, 'model.keras'))
    
    
    return history