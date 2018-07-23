# Objekterkennung in der Datenbank vom oberen Omukuruvaro.
Objektsegmentierung von Wandmalereien des oberen Brandbergs durch TensorFlow und OpenCV

<h3>Mitglieder:</h3>
Tarek Khellaf: xxx<br>
Anita Wichert: xxx<br>
Marvin Busch: xxx<br>
Thomas Oehlke: toehlke@smail.uni-koeln.de<br>

Seiten eines Buches über die Wandmalereien im oberen Brandberg (Namibia) werden durch einen selective search-Algorithmus in verschiedene Regionen unterteilt. Einige dieser Regionen werden anschließend in ein neuronales Netzwerk eingespeist. Wenn dieses Netzwerk erkennt, dass eine Region eine Mensch- oder Tiermalerei ist, wird diese Malerei schließlich durch einen Watershed-Algorithmus segmentiert und umrandet. Diese Umrandung kann anschließend abgespeichert werden.

Auf die erzeugten Bilder könnten weitere Forschungen aufgebaut werden, da die segmentierten Bilder unnütze Informationen, wie Farbflecken, herausgefiltert haben und weitere machine learning Algorithmen somit effizienter arbeiten können.
