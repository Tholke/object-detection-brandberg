#Ein Python Skript, um unsere Pascal Voc Dateien in eine TFRecord datei umzuwandeln.
#TFRecord ist der empfohlene Datentyp von TensorFlow

import numpy as np
import tensorflow as tf
import cv2
import sys
import xmlparser

#Zwei Funktionen, um unsere Daten in ein TensorFlow Feature umzuwandeln
#Diese Funktionen wandelt die Labels um
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#Diese Funktion wandelt die Bilder um
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#Eine Funktion, um mit OpenCV die Pixel unserer Trainingsdaten zu laden
def loadImage(path, obj):
    #Eine gesamte Buchseite wird gelesen
    img = cv2.imread(path)
    #Die bounding box des Trainingsdatums wird ausgeschnitten
    img = img[int(obj[2]):int(obj[4]), int(obj[1]):int(obj[3])]
    #Das Bild wird auf 244x244 Pixel skaliert
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    #Die Farbwerte werden konvertiert
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Das Bild wird in Integer zerlegt
    img = img.astype(np.uint8)
    return img

#Funktion, um die Daten des XML Parsers in labels und Bilddaten zu zerlegen
def preprocess():
    data = xmlparser.getData()
    ys = []
    xs = []

    for page in data:
        #Das erste Objekt jeder Seite ist der Seitenname
        pageName = True
        for object in page:
            x = []
            if pageName:
                pName = object
            else:
                #Wenn das Objekt kein Seitenname ist wird das label in die labelliste (ys) gespeichert
                ys.append(object[0])
                #Für jedes Trainingsdatum wird der Name der Seite und die 4 Ecken der bounding box abgespeichert
                #Die bounding box wird in der Reihenfolge xmin, ymin, xmax, ymax abgespeichert
                x.append(pName)
                x.append(object[1])
                x.append(object[2])
                x.append(object[3])
                x.append(object[4])
            #Manche Trainingsdaten sind leer und werden jetzt rausgefiltert
            if(len(x) > 0):
                #Jedes Trainingsdatum wird in eine Liste aller Trainingsdaten gespeichert
                xs.append(x)
            pageName = False
    #Es wird ein dictionary ausgegeben, dass die xs (Trainingsdaten) und die ys (Labels) beinhaltet
    return {'xs':xs, 'ys':ys }

#Funktion, die die Trainingsdaten und Labels in eine TFRecordsdatei umwandelt
def writeTFRecords(xs, ys):
    #Der Pfad des Ordners, in dem alle Bilder gespeichert sind
    folder = 'BOOK-ZID0824731/images-raw/'
    #Der Pfad, wo die TFRecordsdatei abgespeichert werden soll
    train_filename = 'train.tfrecords'
    
    
    writer = tf.python_io.TFRecordWriter(train_filename)

    index = 0
    for obj in xs:
        #Der Pfad zu dem Bild der Seite, auf der das Objekt zu finden ist
        path = obj[0]
        #Das Bild wird verarbeitet und in Integer umgewandelt
        img = loadImage(folder + path, obj)
        #Das Label wird als '0' gespeichert, wenn es ein Mensch ist und als '1', wenn es ein Tier ist
        label = 0 if 'human' in ys[index] else 1
        #Ein Feature wird erstellt mit label und Bild
        feature = {'train/label': _int64_feature(label),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        print("image", index+1, "of", len(xs))
        index = index+1
        #Ein TFExample wird aus dem Feature erstellt
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        #Das TFExample wird in die TFRecordsdatei geschrieben
        writer.write(example.SerializeToString())

    #Der Output wird geschlossen und das System 
    writer.close()
    #Löscht den Python Outputbuffer
    sys.stdout.flush()
    print("finished")
    
#Steuernde Hauptfunktion
def process():
    prosessed_data = preprocess()
    xs = prosessed_data['xs']
    ys = prosessed_data['ys']
    writeTFRecords(xs, ys)

if __name__ == '__main__':
    process()