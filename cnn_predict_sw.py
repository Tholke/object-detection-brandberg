import tensorflow as tf
import numpy as np
from helpers import pyramid
from helpers import sliding_window
import time
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def cnn_model(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)    
    pool2_flat = tf.reshape(pool2, [-1, 56 * 56 * 64])
    
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training = mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(inputs = dropout, units = 3)
    
    if labels is not None:
        onehot_labels = tf.one_hot(indices = labels, depth = 3)
    
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)})

        
def main(unused_argv):
    saver = tf.train.import_meta_graph('output/brandberg_cnn/model.ckpt-2570.meta')

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('output/brandberg_cnn/'))
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    
        # load the image and define the window width and height
        image = cv2.imread('data/images/BOOK-0824731-0059.jpg')
#         image = cv2.imread('testjpeg.jpg')
        (winW, winH) = (224, 224)
        
        imageList = []
        
        index = 0
        # loop over the image pyramid
        for resized in pyramid(image, scale=1.5):
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in sliding_window(resized, stepSize=30, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                
                index = index + 1

   
                clone = resized.copy()
                clone2 = resized.copy()
                cv2.rectangle(clone2, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                
                cropped = clone[y :y + winW , x : x + winW]
                cropped = cropped.astype(np.float32)
                images = []
                images.append(cropped)
                images = np.asarray(images)
            
                classifier = tf.estimator.Estimator(
                    model_fn=cnn_model, model_dir="output/brandberg_cnn/")
               
                pred_input_fn = tf.estimator.inputs.numpy_input_fn(images, shuffle=False)
        
                classifier = tf.estimator.Estimator(
                    model_fn=cnn_model, model_dir="output/brandberg_cnn/")
            
                input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": images},
                    num_epochs=1,
                    shuffle=False)
            
                predictions = classifier.predict(input_fn = input_fn)
                list_pred = list(predictions)
                
        #         captch_ex('out.png')
                ind = len(list_pred)-1
                for img in reversed(list_pred):
                    probs = img.get('probabilities')
                    #print(probs)
#                     print(str(probs[0]) + "   " + str(probs[1]) + "   " + str(probs[2]))
                    if probs[0] > 0.8 or probs[1] > 0.8:
                        imageList.append([probs[0], probs[1], probs[2], cropped, x, y, winW, winH])
                        print(imageList[-1])
                    ind -= 1
        
                cv2.imshow('Window', images[0])
                cv2.waitKey(1)
                time.sleep(0.025)
    

        print(imageList)        

if __name__ == "__main__":
    tf.app.run()
