import tensorflow as tf
import numpy as np
from helpers import pyramid
from helpers import sliding_window
from helpers import intersection
import watershed
import time
import cv2
import os

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
    saver = tf.train.import_meta_graph('output/brandberg_cnn/model.ckpt-3250.meta')

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('output/brandberg_cnn/'))
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    
        #Ein Bild (eine Buchseite) wird eingelesen und die Fenstergrößé des sliding window definiert
        image = cv2.imread('data/images/BOOK-0824731-0059.jpg')
        (winW, winH) = (224, 224)
        
        #Liste mit einigen Kerninformationen
        imageList = []
        #List ein der später die Koordinaten des gefundenen Bildes gespeichert werden (zum entfernen von Duplikaten)
        debug_list = []
        #Liste der Wahrscheinlichkeit, die das neuronale Netz angegeben hat
        predictionsForSegmentation = []
        
        index = 0
        # Loopt über die Bilder verschiedener Größe
        for resized in pyramid(image, scale=1.5):
            # Loopt über das Bild indem der momentan betrachtete Auschnitt immer wieder verschoben wird
            for (x, y, window) in sliding_window(resized, stepSize=30, windowSize=(winW, winH)):
                # Hat das Fenster nicht die richtige Größe wird es ignoriert
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                
                index = index + 1
                
                # Macht eine Kopie des Bildes
                clone = resized.copy()
                
                # Schneidet den richtigen Auschnitt aus und bereitet es für die Eingabe in Tensorflow vor
                cropped = clone[y :y + winW , x : x + winW]
                cropped = cropped.astype(np.float64)
                images = []
                images.append(cropped)
                images = np.asarray(images)
                
                #Das neuronale Netzwerk wird geladen
                classifier = tf.estimator.Estimator(
                    model_fn=cnn_model, model_dir="output/brandberg_cnn/")
               
                #Regionen werden dem Netzwerk übergeben und es gibt Predictions aus
                input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": images},
                    num_epochs=1,
                    shuffle=False)
            
                predictions = classifier.predict(input_fn = input_fn)
                #Predictions werden in eine Liste gespeichert
                list_pred = list(predictions)
        
                for img in reversed(list_pred):
                    probs = img.get('probabilities')
                
                #Regionen, bei denen sich das Netzwerk zu 95% sicher ist, werden in ein Array gespeichert.
                #D.h. alle Regionen, bei denen das Netzwerk ein Mensch oder ein Tier erkannt hat 
                ind = len(list_pred)-1
                for img in reversed(list_pred):
                    probs = img.get('probabilities')
                    if probs[0] > 0.95 or probs[1] > 0.95:
                        imageList.append([probs[0], probs[1], probs[2], cropped, x, y, winW, winH])
                        predictionsForSegmentation.append(max(probs[0], probs[1]))
                        debug_list.append([x, y, x+winW, y+winH])
#                         print(imageList[-1])
#                         print('Länge:', len(imageList))
                    ind -= 1
        
    i = 0
    arraylen = len(debug_list)
    
    #Durchläuft alle identifizierten Menschen und Tiere und filtert identische Objekte heraus
    #Sind zwei identische Objekte gefunden, wird das mit der niedrigeren Wahrscheinlichkeit gelöscht
    while i < arraylen:
        j = i+1
        while j < arraylen:
            if intersection(debug_list[i], debug_list[j]):
                #Die Wahrscheinlichkeit, mit der das neuronale Netz ein Tier oder Mensch erkannt hat
                probsImgA = predictionsForSegmentation[i]
                probsImgB = predictionsForSegmentation[j]
                
                #Entfernt das Bild mit der geringeren Wahrscheinlichkeit aus allen Listen
                if probsImgB <= probsImgA:
                    imagesForSegmentation.pop(j)
                    debug_list.pop(j)
                    predictionsForSegmentation.pop(j)
                    j -= 1
                    arraylen -= 1
                else:
                    imageList.pop(i)
                    debug_list.pop(i)
                    predictionsForSegmentation.pop(i)
                    i -= 1
                    arraylen -= 1
                    break
            j += 1
        i += 1
            
    
    
    #Pfad unter dem die Segmentierten bilder abgespeichert werden können
    savePath = 'output/'
    
    #Restliche Regionen aus dem zu segmentierenden Array werden durch den Watershed-Algorithmus segmentiert
    for l in imageList:
        saveImagePath = savePath + imagePath[:-4] + '-' + str(i) + '.jpg'
        watershed.computeWatershed(l[3], saveImagePath)

                
                
    print(imageList)        

if __name__ == "__main__":
    tf.app.run()
