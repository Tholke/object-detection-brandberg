# Objekterkennung in der Datenbank vom oberen Omukuruvaro.
Objektsegmentierung von Wandmalereien des oberen Brandbergs durch TensorFlow und OpenCV

<h3>Mitglieder:</h3>
Tarek Khellaf: tkhellaf@smail.uni-koeln.de <br>
Anita Wichert: awicher1@smail.uni-koeln.de <br>
Marvin Busch: mbusch10@smail.uni-koeln.de<br>
Thomas Oehlke: toehlke@smail.uni-koeln.de<br>
<br>Seiten eines Buches über die Wandmalereien im oberen Brandberg (Namibia) werden durch einen selective search-Algorithmus in verschiedene Regionen unterteilt. Einige dieser Regionen werden anschließend in ein neuronales Netzwerk eingespeist. Wenn dieses Netzwerk erkennt, dass eine Region eine Mensch- oder Tiermalerei ist, wird diese Malerei schließlich durch einen Watershed-Algorithmus segmentiert und umrandet. Diese Umrandung kann anschließend abgespeichert werden.

Auf die erzeugten Bilder könnten weitere Forschungen aufgebaut werden, da die segmentierten Bilder unnütze Informationen, wie Farbflecken, herausgefiltert haben und weitere machine learning Algorithmen somit effizienter arbeiten können.

Es gibt zwei verschiedene Ansätze aus denen gewählt werden kann:
1) Selective Search:
  - Mit der Datei createTFRecord.py wird die TFRecord-Datei erstellt
  - Mit der Datei cnn.py wird das neuronale Netz Trainiert
  - Mit der Datei cnn_predict.py wird eine Buchseite bearbeitet, die Objekte erkannt und isoliert
    (Eventuell muss die Zeile "saver = tf.train.import_meta_graph(xxx)" angepasst werden.)
    
2) Sliding Window:
  - Mit der Datei createTFRecord_sw.py wird die TFRecord-Datei erstellt
  - Mit der Datei cnn_sw.py wird das neuronale Netz Trainiert
  - Mit der Datei cnn_predict_sw.py wird eine Buchseite bearbeitet, die Objekte erkannt und isoliert
    (Eventuell muss die Zeile "saver = tf.train.import_meta_graph(xxx)" angepasst werden.)
