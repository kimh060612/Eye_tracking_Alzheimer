import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import parser

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Network Parameters
img_size = 64
n_channel = 3
mask_size = 25

# pathway: eye_left and eye_right
conv1_eye_size = 11
conv1_eye_out = 96
pool1_eye_size = 2
pool1_eye_stride = 2

conv2_eye_size = 5
conv2_eye_out = 256
pool2_eye_size = 2
pool2_eye_stride = 2

conv3_eye_size = 3
conv3_eye_out = 384
pool3_eye_size = 2
pool3_eye_stride = 2

conv4_eye_size = 1
conv4_eye_out = 64
pool4_eye_size = 2
pool4_eye_stride = 2

eye_size = 2 * 2 * 2 * conv4_eye_out

# pathway: face
conv1_face_size = 11
conv1_face_out = 96
pool1_face_size = 2
pool1_face_stride = 2

conv2_face_size = 5
conv2_face_out = 256
pool2_face_size = 2
pool2_face_stride = 2

conv3_face_size = 3
conv3_face_out = 384
pool3_face_size = 2
pool3_face_stride = 2

conv4_face_size = 1
conv4_face_out = 64
pool4_face_size = 2
pool4_face_stride = 2

face_size = 2 * 2 * conv4_face_out

# fc layer
fc_eye_size = 128
fc_face_size = 128
fc_face_mask_size = 256
face_face_mask_size = 128
fc_size = 128
fc2_size = 2


class Itracker:
    def __init__(self):
        self.eye_left = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='eye_left')
        self.eye_right = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='eye_right')
        self.face = tf.placeholder(tf.float32, [None, img_size, img_size, n_channel], name='face')
        self.face_mask = tf.placeholder(tf.float32, [None, mask_size * mask_size], name='face_mask')
        self.y = tf.placeholder(tf.float32, [None, 2], name='pos')
        self.Image_input = np.array([])
        # Store layers weight & bias
        self.weights = {
            'conv1_eye': tf.get_variable('conv1_eye_w', shape=(conv1_eye_size, conv1_eye_size, n_channel, conv1_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv2_eye': tf.get_variable('conv2_eye_w', shape=(conv2_eye_size, conv2_eye_size, conv1_eye_out, conv2_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv3_eye': tf.get_variable('conv3_eye_w', shape=(conv3_eye_size, conv3_eye_size, conv2_eye_out, conv3_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv4_eye': tf.get_variable('conv4_eye_w', shape=(conv4_eye_size, conv4_eye_size, conv3_eye_out, conv4_eye_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv1_face': tf.get_variable('conv1_face_w', shape=(conv1_face_size, conv1_face_size, n_channel, conv1_face_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv2_face': tf.get_variable('conv2_face_w', shape=(conv2_face_size, conv2_face_size, conv1_face_out, conv2_face_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv3_face': tf.get_variable('conv3_face_w', shape=(conv3_face_size, conv3_face_size, conv2_face_out, conv3_face_out), initializer=tf.contrib.layers.xavier_initializer()),
            'conv4_face': tf.get_variable('conv4_face_w', shape=(conv4_face_size, conv4_face_size, conv3_face_out, conv4_face_out), initializer=tf.contrib.layers.xavier_initializer()),
            'fc_eye': tf.get_variable('fc_eye_w', shape=(eye_size, fc_eye_size), initializer=tf.contrib.layers.xavier_initializer()),
            'fc_face': tf.get_variable('fc_face_w', shape=(face_size, fc_face_size), initializer=tf.contrib.layers.xavier_initializer()),
            'fc_face_mask': tf.get_variable('fc_face_mask_w', shape=(mask_size * mask_size, fc_face_mask_size), initializer=tf.contrib.layers.xavier_initializer()),
            'face_face_mask': tf.get_variable('face_face_mask_w', shape=(fc_face_size + fc_face_mask_size, face_face_mask_size), initializer=tf.contrib.layers.xavier_initializer()),
            'fc': tf.get_variable('fc_w', shape=(fc_eye_size + face_face_mask_size, fc_size), initializer=tf.contrib.layers.xavier_initializer()),
            'fc2': tf.get_variable('fc2_w', shape=(fc_size, fc2_size), initializer=tf.contrib.layers.xavier_initializer())
        }
        self.biases = {
            'conv1_eye': tf.Variable(tf.constant(0.1, shape=[conv1_eye_out])),
            'conv2_eye': tf.Variable(tf.constant(0.1, shape=[conv2_eye_out])),
            'conv3_eye': tf.Variable(tf.constant(0.1, shape=[conv3_eye_out])),
            'conv4_eye': tf.Variable(tf.constant(0.1, shape=[conv4_eye_out])),
            'conv1_face': tf.Variable(tf.constant(0.1, shape=[conv1_face_out])),
            'conv2_face': tf.Variable(tf.constant(0.1, shape=[conv2_face_out])),
            'conv3_face': tf.Variable(tf.constant(0.1, shape=[conv3_face_out])),
            'conv4_face': tf.Variable(tf.constant(0.1, shape=[conv4_face_out])),
            'fc_eye': tf.Variable(tf.constant(0.1, shape=[fc_eye_size])),
            'fc_face': tf.Variable(tf.constant(0.1, shape=[fc_face_size])),
            'fc_face_mask': tf.Variable(tf.constant(0.1, shape=[fc_face_mask_size])),
            'face_face_mask': tf.Variable(tf.constant(0.1, shape=[face_face_mask_size])),
            'fc': tf.Variable(tf.constant(0.1, shape=[fc_size])),
            'fc2': tf.Variable(tf.constant(0.1, shape=[fc2_size]))
        }

        # Construct model
        self.pred = self.itracker_nets(self.eye_left, self.eye_right, self.face, self.face_mask, self.weights, self.biases)

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k, strides):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                              padding='VALID')

    # Create model
    def itracker_nets(self, eye_left, eye_right, face, face_mask, weights, biases):
        # pathway: left eye
        eye_left = self.conv2d(eye_left, weights['conv1_eye'], biases['conv1_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool1_eye_size, strides=pool1_eye_stride)

        eye_left = self.conv2d(eye_left, weights['conv2_eye'], biases['conv2_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool2_eye_size, strides=pool2_eye_stride)

        eye_left = self.conv2d(eye_left, weights['conv3_eye'], biases['conv3_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool3_eye_size, strides=pool3_eye_stride)

        eye_left = self.conv2d(eye_left, weights['conv4_eye'], biases['conv4_eye'], strides=1)
        eye_left = self.maxpool2d(eye_left, k=pool4_eye_size, strides=pool4_eye_stride)

        # pathway: right eye
        eye_right = self.conv2d(eye_right, weights['conv1_eye'], biases['conv1_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool1_eye_size, strides=pool1_eye_stride)

        eye_right = self.conv2d(eye_right, weights['conv2_eye'], biases['conv2_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool2_eye_size, strides=pool2_eye_stride)

        eye_right = self.conv2d(eye_right, weights['conv3_eye'], biases['conv3_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool3_eye_size, strides=pool3_eye_stride)

        eye_right = self.conv2d(eye_right, weights['conv4_eye'], biases['conv4_eye'], strides=1)
        eye_right = self.maxpool2d(eye_right, k=pool4_eye_size, strides=pool4_eye_stride)

        # pathway: face
        face = self.conv2d(face, weights['conv1_face'], biases['conv1_face'], strides=1)
        face = self.maxpool2d(face, k=pool1_face_size, strides=pool1_face_stride)

        face = self.conv2d(face, weights['conv2_face'], biases['conv2_face'], strides=1)
        face = self.maxpool2d(face, k=pool2_face_size, strides=pool2_face_stride)

        face = self.conv2d(face, weights['conv3_face'], biases['conv3_face'], strides=1)
        face = self.maxpool2d(face, k=pool3_face_size, strides=pool3_face_stride)

        face = self.conv2d(face, weights['conv4_face'], biases['conv4_face'], strides=1)
        face = self.maxpool2d(face, k=pool4_face_size, strides=pool4_face_stride)

        # fc layer
        # eye
        eye_left = tf.reshape(eye_left, [-1, int(np.prod(eye_left.get_shape()[1:]))])
        eye_right = tf.reshape(eye_right, [-1, int(np.prod(eye_right.get_shape()[1:]))])
        eye = tf.concat([eye_left, eye_right], 1)
        eye = tf.nn.relu(tf.add(tf.matmul(eye, weights['fc_eye']), biases['fc_eye']))

        # face
        face = tf.reshape(face, [-1, int(np.prod(face.get_shape()[1:]))])
        face = tf.nn.relu(tf.add(tf.matmul(face, weights['fc_face']), biases['fc_face']))

        # face mask
        face_mask = tf.nn.relu(tf.add(tf.matmul(face_mask, weights['fc_face_mask']), biases['fc_face_mask']))

        face_face_mask = tf.concat([face, face_mask], 1)
        face_face_mask = tf.nn.relu(tf.add(tf.matmul(face_face_mask, weights['face_face_mask']), biases['face_face_mask']))

        # all
        fc = tf.concat([eye, face_face_mask], 1)
        fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['fc']), biases['fc']))
        out = tf.add(tf.matmul(fc, weights['fc2']), biases['fc2'])
        return out

    def Tracker(self, Image):
        self.Image_input = Image
        
        


if __name__ == "__main__":
    

    
    pass