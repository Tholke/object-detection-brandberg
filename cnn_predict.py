#Dieses Skript kann genutzt werden, um dem CNN ein Bild zum verarbeiten zu geben und die interessanten Regionen zu segmentieren

import tensorflow as tf
import numpy as np
import cv2
import os
import watershed

#Unterdrückt eine Fehlermeldung bei MacOS
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
    saver = tf.train.import_meta_graph('tmp/brandberg_cnn/model.ckpt-2056.meta')

    with tf.Session() as sess:
        #Der letzte Checkpoint des Models wird geladen und globale und lokale TensorFlowVariablen initialisiert
        saver.restore(sess, tf.train.latest_checkpoint('tmp/brandberg_cnn/'))
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
    #Ein Bild (eine Buchseite) wird eingelesen
    dirPath = '/Users/thomas/Desktop/Uni/SoSe18/KI/BOOK-ZID0824731/images-raw/'
    imagePath = 'BOOK-0824731-0058.jpg'
    img = cv2.imread(dirPath + imagePath)
    
    #Eine Kopie des Bildes wird auf 600 Pixel höhe skaliert, das Seitenverhältnis wird beibehalten
    #Dadurch soll der Rechenaufwand des selective search-Algorithmus verringert werden
    newWidth = int(img.shape[1]*600/img.shape[0])
    resizedImg = cv2.resize(img, (newWidth, 600))
    
    #Selective search wird erstellt und ausgeführt. Schnelle Variante wird gewählt, um den Rechenaufwand weiter zu verringern
    selectiveSearch = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selectiveSearch.setBaseImage(resizedImg)
    selectiveSearch.switchToSelectiveSearchFast()
    rects = selectiveSearch.process()
    print('Anzahl gefundener Regionen:',len(rects))

    #Einige Regionen werden von vornherein rausgefiltert
    rectsToDelete = []
    
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        area = w * h
        
        #Regionen sollen gefiltert werden, wenn die Fläche über 90 000 oder unter 100 Pixel groß ist oder die Höhe oder Breite unter 50 Pixel
        if area > 90000 or area < 100 or w < 50 or h < 50:
            rectsToDelete.append(i)
    
    print(len(rectsToDelete), 'gefiltert.', len(rects)-len(rectsToDelete), 'Regionen werden übergeben')

    #Regionen die den genannten Kriterien entsprechen werden gelöscht
    for j in reversed(rectsToDelete):
        rects = np.delete(rects, j, 0)

    imagesToClassify = []
    imagesForSegmentation = []
    hscale = img.shape[0] / resizedImg.shape[0]
    wscale = img.shape[1] / resizedImg.shape[1]
    
    #Übrige Regionen werden aus der Buchseite ausgeschnitten und in einen Array gespeichert
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        #Regionen werden skaliert und aus dem Originalbild ausgeschnitten (nicht aus dem 600x600 Pixel Bild)
        rectImg = img[int(y*hscale):int((y+h)*hscale), int(x*wscale):int((x+w)*wscale)]
        #Regionen werden in einen Array gespeichert mit dem später die Segmentation durchgeführt wird
        imagesForSegmentation.append(rectImg)
        #Region wird für das neuronale Netzwerk vorbereitet und in einen anderen Array gespeichert
        rectImg = cv2.resize(rectImg, (224, 224), interpolation=cv2.INTER_CUBIC)
        rectImg = cv2.cvtColor(rectImg, cv2.COLOR_BGR2RGB)
        rectImg = rectImg.astype(np.float32)
        imagesToClassify.append(rectImg)

    #Prediction-Array wird für das neuronale Netzwerk zum NumPy-Array umgewandelt
    imagesToClassify = np.asarray(imagesToClassify)

    #Das neuronale Netzwerk wird geladen
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model, model_dir="/Users/thomas/Desktop/Uni/SoSe18/KI/tmp/brandberg_cnn")

    #Regionen werden dem Netzwerk übergeben und es gibt Predictions aus
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": imagesToClassify},
        num_epochs=1,
        shuffle=False)
    
    predictions = classifier.predict(input_fn = input_fn)
    #Predictions werden in eine Liste gespeichert
    list_pred = list(predictions)
    
    #Regionen, bei denen sich das Netzwerk nicht zu 95% sicher ist, werden aus dem Array für die Segmentation gelöscht
    #D.h. alle Regionen, bei denen das Netzwerk kein Mensch oder Tier erkannt hat 
    ind = len(list_pred)-1
    deletedImages = 0
    for img in reversed(list_pred):
        probs = img.get('probabilities')
        if probs[0] < 0.95 and probs[1] < 0.95:
            imagesForSegmentation.pop(ind)
            deletedImages += 1
        ind -= 1
    print(len(imagesForSegmentation), 'Bilder werden segmentiert.')
    
    #Pfad unter dem die Segmentierten bilder abgespeichert werden können
    savePath = '/Users/thomas/Desktop/Uni/SoSe18/KI/BOOK-ZID0824731/images-saved/'
    
    #Restliche Regionen aus dem zu segmentierenden Array werden durch den Watershed-Algorithmus segmentiert
    for i, image in enumerate(imagesForSegmentation):
        saveImagePath = savePath + imagePath[:-4] + '-' + str(i) + '.jpg'
        watershed.computeWatershed(image, saveImagePath)

if __name__ == "__main__":
    tf.app.run()