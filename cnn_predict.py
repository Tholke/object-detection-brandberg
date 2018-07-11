import tensorflow as tf
import numpy as np
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
    
    logits = tf.layers.dense(inputs = dropout, units = 2)
    
    onehot_labels = tf.one_hot(indices = labels, depth = 2)
    
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        
def main(unused_argv):
    saver = tf.train.import_meta_graph('tmp/brandberg_cnn/model.ckpt-1.meta')

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('tmp/brandberg_cnn/'))
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
    
    img = cv2.imread('testjpg.jpeg')

    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
  
    images = []

    images.append(img)
    images = np.asarray(list(images))

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model, model_dir="/Users/thomas/Desktop/Uni/SoSe18/KI/tmp/brandberg_cnn")

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": images},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn = input_fn)
    
    print(list(predictions))

if __name__ == "__main__":
    tf.app.run()