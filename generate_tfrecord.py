"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python2 generate_tfrecord.py --csv_input=data/ocr_train_labels.csv  --output_path=train.record
 
  # Create test data:
  python2 generate_tfrecord.py --csv_input=data/ocr_test_labels.csv  --output_path=test.record
  
  
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import json

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

## defining different variables
data_dir = '../data/NIST19/'
img_size = 128
flatten_image_size = img_size**2
num_channels = 3
img_shape = (img_size,img_size,num_channels)
num_classes = 62 # 26 lower case alphabets, 26 upper case alphabets, 10 digits

# reading the maps
label_cls_name_map = {}
label_name_cls_map = {}
with open('ocr_checkpoints/label_cls_name.json', 'r') as f:
    label_cls_name_map = json.loads(f.read())
    
for k,v in label_cls_name_map.iteritems():
    label_name_cls_map[v] = k

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label:
        return int(label_name_cls_map[row_label])+1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

count_=0
def create_tf_example(group, path):
    dir_ = group.filename.split('_')[1]
    with tf.gfile.GFile(os.path.join(path, '{}/train_{}/{}'.format(dir_, dir_, group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    global count_
    print(count_ , ' ', group.filename)
    count_+=1
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    #path = os.path.join(os.getcwd(), 'images')
    path = data_dir
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
