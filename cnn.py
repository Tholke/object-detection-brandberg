#Dieses Skript erstellt ein convoluional Neural Network und trainiert es

import numpy as np
import tensorflow as tf
import os
import cv2

#Unterdrückt eine Fehlermeldung bei MacOS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Funktion, die eine Bild-Label-Kombination lädt
def read_decode(filename_queue):
    #Der Reader wird erstellt
    reader = tf.TFRecordReader()
    #Eine Bild-Label-Kombination wird gelesen
    _, serialized_example = reader.read(filename_queue)
    feature = {'label': tf.FixedLenFeature([], tf.int64),
               'image': tf.FixedLenFeature([], tf.string)}
    
    #Bild-Label-Kombination wird geparsed, das Bild in 32Bit Floats umgewandelt und die Labels in Integer
    features = tf.parse_single_example(serialized_example, features = feature)
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int32)
  
    return image, label

#Parset die gesamte TFRecords Datei und gibt Features und Labels aus 
def read_tfrecords(path):
    features = []
    labels = []
    with tf.Session() as session:
        
        filename_queue = tf.train.string_input_producer([ path ], num_epochs = 1)
        
        #Lädt eine Bild-Label-Kombination und setzt die Größe des Bilds auf 224x224 Pixel
        image, label = read_decode(filename_queue)
        image = tf.reshape(image, [224, 224, 3])
        
        #Initialisiert globale und lokale TensorFlowVariablen
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        session.run(init_op)
        
        #Initialisiert einen Coordinator und öffnet mehrere Threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        try:
            #Lädt alle Bilder und Labels und speichert sie in Arrays
            while True:
                img, l = session.run([image, label])
                features.append(img)
                labels.append(l)
        
        #Führt den Try-Block aus, bis das Ende der TFRecords-Datei erreicht ist
        except tf.errors.OutOfRangeError:
            coord.request_stop()
        
        #Führt schließlich alle Threads zusammen und beendet den Coordinator
        finally:
            coord.request_stop()
            coord.join(threads)
        
    return features, labels

#Erstellt das neuronale Netzwerk
def cnn_model(features, labels, mode):
    #Bilder werden (erneut) auf 224x224 Pixel angepasst
    input_layer = tf.reshape(features["x"], [-1, 224, 224, 3])
    
    #Erster convolutional Layer
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    
    #Zweiter convolutional Layer
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)    
    #Output des zweiten convolutional Layers wird für den Denselayer geglättet
    pool2_flat = tf.reshape(pool2, [-1, 56 * 56 * 64])
    
    #Erster vollständig verbundener Layer mit Dropout, um Overfitting entgegenzuwirken
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training = mode == tf.estimator.ModeKeys.TRAIN)
    
    #Outputlayer
    logits = tf.layers.dense(inputs = dropout, units = 3)
    
    if labels is not None:
        #Labels werden Onehot kodiert
        onehot_labels = tf.one_hot(indices = labels, depth = 3)
    
        #Loss wird berechnet
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    
    #Predictions besitzen eine Klasse und eine Wahrscheinlichkeit
    predictions = {
      #Der höchste Output wird als Klasse gewählt
      "classes": tf.argmax(input=logits, axis=1),
      #Wahrscheinlichkeit wird durch Softmax angegeben
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    #Im Prediction-Modus werden Predictions ausgegeben
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)})

    #Im Trainings-Modus wird das neuronale Netzwerk trainiert
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #Das neuronale Netzwerk wird evaluiert
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    #Trainingsdaten werden eingelesen und in NumPy-Arrays umgewandelt
    train_path = 'train.tfrecords'
    features, labels = read_tfrecords(train_path)
    features = np.asarray(features)
    labels = np.asarray(labels)
    
    #Evaluierungsdaten werden eingelesen und in NumPy-Arrays umgewandelt
    eval_path = 'eval.tfrecords'
    eval_features, eval_labels = read_tfrecords(eval_path)
    eval_features = np.asarray(eval_features)
    eval_labels = np.asarray(eval_labels)

    #Das neuronale Netzwerk wird geladen
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model, model_dir="/Users/thomas/Desktop/Uni/SoSe18/KI/tmp/brandberg_cnn")

    #Zu speichernde Daten
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    
    #Trainingsdaten werden dem Netzwerk übergeben und es wird trainiert
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": features},
      y=labels,
      batch_size=1,
      num_epochs=1,
      shuffle=True)

    classifier.train(
      input_fn=train_input_fn,
      hooks=[logging_hook])
    
    #Evaluierungsdaten werden dem Netzwerk übergeben und es wird evaluiert
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_features},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
    
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    
if __name__ == "__main__":
    tf.app.run()