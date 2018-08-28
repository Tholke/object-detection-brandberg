#Dieses Skript kann genutzt werden, um dem CNN ein Bild zum verarbeiten zu geben und die interessanten Regionen zu segmentieren

import tensorflow as tf
import numpy as np
import cv2
import os
import watershed
from helpers import intersection
from helpers import pyramid
from helpers import sliding_window

#Unterdrückt eine Warnmeldung bei MacOS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Das ganze Model muss neu angegeben werden. Auch wenn es schon vorher trainiert wurde
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
    
def main(unused_argv):
    #Das trainierte Model wird geladen
    saver = tf.train.import_meta_graph('tmp/brandberg_cnn/model.ckpt-3021.meta')

    with tf.Session() as sess:
        #Der letzte Checkpoint des Models wird geladen und globale und lokale TensorFlowVariablen initialisiert
        saver.restore(sess, tf.train.latest_checkpoint('tmp/brandberg_cnn/'))
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
    #Ein Bild (eine Buchseite) wird eingelesen
    dirPath = 'data/images/'
    imagePath = 'BOOK-0824731-0119.jpg'
    image = cv2.imread(dirPath + imagePath)
    
    #Bild kann zur Erhöhung der Rechengeschwindigkeit und zur Reduktion von nötigem Arbeitsspeicher um die Hälfte (oder mehr) verkleinert werden, rediuziert jedoch die Genauigkeit
        #image = cv2.resize(image, (int(image.shape[0]/2), int(image.shape[1]/2)), interpolation=cv2.INTER_CUBIC)
    
    (winW, winH) = (224, 224)
    imagesToClassify = []
    imagesForSegmentation = []
    scaledBoundingBoxes = []

    index = 0
    # Loopt über die Bilder verschiedener Größe
    for resized in pyramid(image, scale=1.5):
        
        # Loopt über das Bild indem der momentan betrachtete Auschnitt immer wieder verschoben wird
        for (x, y, window) in sliding_window(resized, stepSize=30, windowSize=(winW, winH)):
            # Hat das Fenster nicht die richtige Größe wird es ignoriert
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            scaledBoundingBoxes.append([x, y, x + winW, y + winH])
            imagesForSegmentation.append(window)
            #Region wird für das neuronale Netzwerk vorbereitet und in einen anderen Array gespeichert
            window = cv2.resize(window, (224, 224), interpolation=cv2.INTER_CUBIC)
            window = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
            window = window.astype(np.float32)
            imagesToClassify.append(window)

    #Prediction-Array wird für das neuronale Netzwerk zum NumPy-Array umgewandelt
    imagesToClassify = np.asarray(imagesToClassify)

    #Das neuronale Netzwerk wird geladen
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model, model_dir="tmp/brandberg_cnn")

    #Regionen werden dem Netzwerk übergeben und es gibt Predictions aus
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": imagesToClassify},
        num_epochs=1,
        shuffle=False)
    
    predictions = classifier.predict(input_fn = input_fn)
    #Predictions werden in eine Liste gespeichert
    list_pred = list(predictions)
    print(list_pred)
    
    #Regionen, bei denen sich das Netzwerk nicht zu 99% sicher ist, werden aus dem Array für die Segmentation gelöscht
    #D.h. alle Regionen, bei denen das Netzwerk kein Mensch oder Tier erkannt hat 
    ind = len(list_pred)-1
    deletedImages = 0
    predictionsForSegmentation = []
    for img in reversed(list_pred):
        probs = img.get('probabilities')
        if probs[0] < 0.99 and probs[1] < 0.99  :
            imagesForSegmentation.pop(ind)
            scaledBoundingBoxes.pop(ind)
            deletedImages += 1
        else:
            predictionsForSegmentation.append(max(probs[0], probs[1]))
        ind -= 1
    
    i = 0
    arraylen = len(scaledBoundingBoxes)
    
    #Durchläuft alle identifizierten Menschen und Tiere und filtert identische Objekte heraus
    #Sind zwei identische Objekte gefunden, wird das mit der niedrigeren Wahrscheinlichkeit gelöscht
    while i < arraylen:
        j = i+1
        while j < arraylen:
            if intersection(scaledBoundingBoxes[i], scaledBoundingBoxes[j]):
                #Die Wahrscheinlichkeit, mit der das neuronale Netz ein Tier oder Mensch erkannt hat
                probsImgA = predictionsForSegmentation[i]
                probsImgB = predictionsForSegmentation[j]
                
                #Entfernt das Bild mit der geringeren Wahrscheinlichkeit aus allen Listen
                if probsImgB <= probsImgA:
                    imagesForSegmentation.pop(j)
                    scaledBoundingBoxes.pop(j)
                    j -= 1
                    arraylen -= 1
                else:
                    imagesForSegmentation.pop(i)
                    scaledBoundingBoxes.pop(i)
                    i -= 1
                    arraylen -= 1
                    break
            j += 1
        i += 1
            
    print(len(imagesForSegmentation), 'Bilder werden segmentiert.')
    
    #Pfad unter dem die Segmentierten bilder abgespeichert werden können
    savePath = 'output/'
    
    #Restliche Regionen aus dem zu segmentierenden Array werden durch den Watershed-Algorithmus segmentiert
    for i, image in enumerate(imagesForSegmentation):
        saveImagePath = savePath + imagePath[:-4] + '-' + str(i) + '.jpg'
        watershed.computeWatershed(image, saveImagePath)

if __name__ == "__main__":
    tf.app.run()
