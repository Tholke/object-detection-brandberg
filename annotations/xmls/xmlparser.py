# Ein Parser, um die Daten aus den Pascal VOC Dateien zu extrahieren
# Daten werden in einer Liste gespeichert. 
# Jede XML Datei hat eine eigene Liste mit dem .jpg-Dateinamen und der annotation und der bounding box aller Objekte.
# Das Python Skript muss im gleichen Ordner wie die XML Dateien ausgeführt werden.

import xml.etree.ElementTree as ET
import os
import glob

#Eine Hauptliste für alle XML Dateien
data = []

for file in glob.glob('*.xml'):
    #XML Datei wird geparsed
    tree = ET.parse(file)
    root = tree.getroot()

    #Eine Liste für die geparste XML Datei
    xmldata = []
    #In die Liste wird der Dateiname eingefügt
    xmldata.append(root.find('filename').text)

    objects = list(root.findall('object'))

    for obj in objects:
        #Eine Liste für jedes Objekt
        pobject = []
        #Die annotation wird in die Liste gespeichert
        pobject.append(obj.find('name').text)
        bbox = obj.find('bndbox')
        for point in bbox:
            #Alle bounding box Punkte werden in die Liste gespeichert
            pobject.append(point.text)
        #Das Objekt wird in die Liste für die XML Datei gespeichert
        xmldata.append(pobject)

    #Die Liste für die XML Datei wird in die Hauptliste gespeichert
    data.append(xmldata)