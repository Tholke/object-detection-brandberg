#Ein Python Skript, um unsere Pascal Voc Dateien in eine TFRecord datei umzuwandeln.
#TFRecord ist der empfohlene Datentyp von TensorFlow

import numpy as np
import tensorflow as tf
import cv2
import sys
import xmlparser
from random import shuffle
from math import floor
from random import randint

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
    #Das Bild wird auf 224x224 Pixel skaliert
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    #Die Farbwerte werden konvertiert
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #Das Bild wird in Integer zerlegt
    img = img.astype(np.uint8)
    return img

#Funktion, um die Daten des XML Parsers in labels und Bilddaten zu zerlegen und True-negatives zu laden
def preprocess():
    #Lädt die Daten der Pascal Voc Dateien als Array
    data = xmlparser.getData()
    #In ys werden die labels gespeichert
    ys = []
    #In xs werden die bounding boxes gespeichert
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
     
    #Datei mit Bildern für True-Negatives wird geladen
    fname = 'truenegatives.txt'   
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]    
    
    tn_no = 0
    for line in content:
        print("BOOK-ZID0824731/images-raw/{}".format(line))
        #Bild wird gelesen
        img = cv2.imread("BOOK-ZID0824731/images-raw/{}".format(line))
        height, width, _ = img.shape
               
        x = []
        pName = line
        #30 zufällige Regionen (224x224 Pixel) werden aus dem Bild ausgeschnitten
        for i in range(30): 
            xmin = randint(0,width-224)
            ymin = randint(0,height-224)
            xmax = xmin + 224
            ymax = ymin + 224
            
            #Label wird in die Labelliste gespeichert
            ys.append('negative')
            
            #Für jedes Ture Negative wird der Name der Seite und die 4 Ecken der bounding box abgespeichert
            #Die bounding box wird in der Reihenfolge xmin, ymin, xmax, ymax abgespeichert
            x.append(pName)
            x.append(xmin)
            x.append(ymin)
            x.append(xmax)
            x.append(ymax)
            
            #Jedes Trainingsdatum wird in eine Liste aller Trainingsdaten gespeichert
            xs.append(x)
            print("Adding true negative no. " + (str(tn_no)))
            tn_no = tn_no + 1        
  
          
    #Es wird ein dictionary ausgegeben, dass die xs (Trainingsdaten) und die ys (Labels) beinhaltet
    return {'xs':xs, 'ys':ys }

#Funktion, die die Trainingsdaten und Labels in eine TFRecordsdatei umwandelt und True-Negatives erstellt
def writeTFRecords(xs, ys, filename):
    #Der Pfad des Ordners, in dem alle Bilder gespeichert sind
    folder = 'BOOK-ZID0824731/images-raw/'
    
    #TFWriter wird erstellt
    writer = tf.python_io.TFRecordWriter(filename)

    #Zählvariable für print-Informationen
    index = 0
    
    for obj in xs:
        #Der Pfad zu dem Bild der Seite, auf der das Objekt zu finden ist
        path = obj[0]
        print("path:", path)
        #Das Bild wird verarbeitet und in Integer umgewandelt
        img = loadImage(folder + path, obj)
        #Das Label wird als '0' gespeichert, wenn es ein Mensch ist und als '1', wenn es ein Tier ist
        label = 0 if 'human' in ys[index] else 1 if 'animal' in ys[index] else 2
        #Ein Feature wird erstellt mit label und Bild
        feature = {'label': _int64_feature(label),
                   'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

        print("image", index+1, "of", len(xs), "(", ((index+1)/len(xs))*100, "%)", "in:", filename)

        #Ein TFExample wird aus dem Feature erstellt
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        #Das TFExample wird in die TFRecordsdatei geschrieben
        writer.write(example.SerializeToString())
         
        index += 1
    
    #Der Output wird geschlossen und das System 
    writer.close()
    #Löscht den Python Outputbuffer
    sys.stdout.flush()
    print("finished")
    
#Hauptfunktion
def process():
    #Der Pfad, wo die TFRecordsdatei für die Trainings- und Evaluierungsdaten abgespeichert werden sollen
    train_filename = 'train.tfrecords'
    eval_filename = 'eval.tfrecords'
    
    prosessed_data = preprocess()
    xs = prosessed_data['xs']
    ys = prosessed_data['ys']
    
    #20 Prozent der annotations werden zu Evaluierungsdaten
    eval_length = floor(len(xs) * 0.2)
    
    #Bilder und Labels werden in Arrays geladen
    eval_xs = xs[0:eval_length]
    train_xs = xs[eval_length:]
    
    eval_ys = ys[0:eval_length]
    train_ys = ys[eval_length:]
    
    #Die TFRecords Dateien werden geschrieben und gespeichert
    writeTFRecords(train_xs, train_ys, train_filename)
    writeTFRecords(eval_xs, eval_ys, eval_filename)

if __name__ == '__main__':
    process()