import numpy as np
import tensorflow as tf
import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def read_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feature = {'train/label': tf.FixedLenFeature([], tf.int64),
               'train/image': tf.FixedLenFeature([], tf.string)}
    
    features = tf.parse_single_example(serialized_example, features = feature)
    image = tf.decode_raw(features['train/image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['train/label'], tf.int32)
  
    return image, label

def read_tfrecords(path):
    index = 0
    features = []
    labels = []
    with tf.Session() as session:  
        filename_queue = tf.train.string_input_producer([ path ], num_epochs = 1)
        
        image, label = read_decode(filename_queue)
        image = tf.reshape(image, [224, 224, 3])
        
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        session.run(init_op)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        try:
            while True:
                img, l = session.run([image, label])
#                cv2.imshow("l",img)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
                features.append(img)
                labels.append(l)
                index += 1
                    
        except tf.errors.OutOfRangeError:
            coord.request_stop()
            
        finally:
            coord.request_stop()
            coord.join(threads)

        
    return features, labels

def cnn_model(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])
    #Output Tensor Shape: [1, 244, 244, 32]
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
    #Output Tensor Shape: [1, 122, 122, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    #Output Tensor Shape: [1, 122, 122, 64]
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    #Output Tensor Shape: [1, 61, 61, 64]
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


    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    path = 'train.tfrecords'
    features, labels = read_tfrecords(path)

    features = np.asarray(features)
    labels = np.asarray(labels)

    classifier = tf.estimator.Estimator(
        model_fn=cnn_model, model_dir="/Users/thomas/Desktop/Uni/SoSe18/KI/tmp/brandberg_cnn")
    
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": features},
      y=labels,
      batch_size=1,
      num_epochs=1,
      shuffle=True)

    classifier.train(
      input_fn=train_input_fn,
      hooks=[logging_hook])
    
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": features},
      y=labels,
      num_epochs=1,
      shuffle=False)
    
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    
if __name__ == "__main__":
    tf.app.run()