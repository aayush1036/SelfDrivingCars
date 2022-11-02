import torchvision
import pandas as pd 
from tqdm import tqdm 
import os 
import yaml

TRAIN_LABELS = os.path.join('bdd100k','labels', 'bdd100k_labels_images_train.json')
VAL_LABELS = os.path.join('bdd100k','labels', 'bdd100k_labels_images_val.json')
TRAIN_IMAGES = os.path.join('bdd100k','images','100k','train')
VAL_IMAGES = os.path.join('bdd100k','images','100k', 'val')
TRAIN_SIZE = 20000
VAL_SIZE = 2000
IMG_HEIGHT = 720
IMG_WIDTH = 1280
TRAIN_DESTINATION = os.path.join('Data', 'train')
VAL_DESTINATION = os.path.join('Data', 'val')

train_labels = pd.read_json(TRAIN_LABELS)
val_labels = pd.read_json(VAL_LABELS)

train_images = os.listdir(TRAIN_IMAGES)
val_images = os.listdir(VAL_IMAGES)
train_path = [os.path.join(TRAIN_IMAGES, x) for x in train_images]
val_path = [os.path.join(VAL_IMAGES, x) for x in val_images]

train_source_dict = dict(zip(train_images, train_path))
val_source_dict = dict(zip(val_images, val_path))

train_labels['source'] = train_labels['name'].map(train_source_dict)
val_labels['source'] = val_labels['name'].map(val_source_dict)

train_sample = train_labels.sample(TRAIN_SIZE)
val_sample = val_labels.sample(VAL_SIZE)

# Source: https://github.com/ucbdrive/bdd100k/blob/master/doc/format.md#bdd100k-details
labels = [
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck"
]
labelmap = dict(zip(labels, range(len(labels))))

assert train_sample.isnull().sum().sum() == 0 and val_sample.isnull().sum().sum() == 0

for idx, label in tqdm(train_sample.iterrows(), total=TRAIN_SIZE):
    imgname = label['name']
    img_path = os.path.join(TRAIN_DESTINATION, imgname)
    img_id = imgname.split('.')[0]
    img = torchvision.io.read_image(label['source'])
    torchvision.io.write_jpeg(img,img_path)
    line_text = []
    for l in label['labels']:
        if 'box2d' in l.keys():
            object_name = l['category']
            object_name = labelmap[object_name]
            xcenter = float(l['box2d']['x1'] + l['box2d']['x2'])/2
            ycenter = float(l['box2d']['y1'] + l['box2d']['y2'])/2
            width = float(l['box2d']['x2'] - l['box2d']['x1']) 
            height = float(l['box2d']['y2'] - l['box2d']['y1'])
            
            xcenter = xcenter/IMG_WIDTH
            ycenter = ycenter/IMG_HEIGHT
            width = width/IMG_WIDTH
            height = height/IMG_HEIGHT
            
            annotation_text = f'{object_name} {xcenter} {ycenter} {width} {height}'
            line_text.append(annotation_text)
        with open(os.path.join(TRAIN_DESTINATION, f'{img_id}.txt'), 'w') as f:
            f.write('\n'.join(line_text))  
            
for idx, label in tqdm(val_sample.iterrows(), total=VAL_SIZE):
    imgname = label['name']
    img_path = os.path.join(VAL_DESTINATION, imgname)
    img_id = imgname.split('.')[0]
    img = torchvision.io.read_image(label['source'])
    torchvision.io.write_jpeg(img,img_path)
    line_text = []
    for l in label['labels']:
        if 'box2d' in l.keys():
            object_name = l['category']
            object_name = labelmap[object_name]
            xcenter = float(l['box2d']['x1'] + l['box2d']['x2'])/2
            ycenter = float(l['box2d']['y1'] + l['box2d']['y2'])/2
            width = float(l['box2d']['x2'] - l['box2d']['x1']) 
            height = float(l['box2d']['y2'] - l['box2d']['y1'])
            
            xcenter = xcenter/IMG_WIDTH
            ycenter = ycenter/IMG_HEIGHT
            width = width/IMG_WIDTH
            height = height/IMG_HEIGHT
            
            annotation_text = f'{object_name} {xcenter} {ycenter} {width} {height}'
            line_text.append(annotation_text)
        with open(os.path.join(VAL_DESTINATION, f'{img_id}.txt'), 'w') as f:
            f.write('\n'.join(line_text))  
            
data = {
    'train':os.path.abspath(TRAIN_DESTINATION),
    'val':os.path.abspath(VAL_DESTINATION),
    'names':list(labelmap.keys()),
    'nc':len(labelmap)
}

with open('data.yaml', 'w') as f: 
    yaml.safe_dump(data, f)