import cv2

#Eine Funktion zum resizen von Bildern
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	#Speichert die Höhe und Breite des Bildes
	dim = None
	(h, w) = image.shape[:2]

	#Sollte die Höhe und Breite 'None' sein, wird das Originalbild zurückgegeben
	if width is None and height is None:
		return image

	#Überprüft ob die Breite 'None' ist
	if width is None:
		#Errechnet das Verhältnis der Höhe und die Dimensionen
		r = height / float(h)
		dim = (int(w * r), height)

	#Überprüft ob die Höhe 'None' ist
	else:
		#Errechnet das Verhältnis der Höhe und die Dimensionen
		r = width / float(w)
		dim = (width, int(h * r))

	#Resized das Bild entsprechend der errechneten Dimensionen und gibt es zurück
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized

#Die Funktion, mit der das Gesamtbild für das sliding window mehrmals verkleinert wird
def pyramid(image, scale=1.5, minSize=(30, 30)):
	#Gibt das das Originalbild zurück
	yield image

	#Loopt durch die Bildpyramide
	while True:
		#Errechnet die neuen Dimensionen und passt das Bild entsprechend an
		w = int(image.shape[1] / scale)
		image = resize(image, width=w)

		#Ist das Bild zu klein stoppt der Vorgang
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		#Gibt das nächst kleinere Bild der Pyramide zurück
		yield image

#Das sliding window
def sliding_window(image, stepSize, windowSize):
	#Das sliding window geht durch das Bild
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			#Gibt den momentanen Bildausschnitt zurück
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

#Gibt True an, wenn sich zwei Bounding-Boxes zu viel überschneiden (da wahrscheinlich selbes Motiv)
def intersection(boxA, boxB):
	#Speichert die Koordinaten der Überschneidungsfläche (die Koordinaten des Punktes links oben und des Punktes rechts unten)
	interXmin = max(boxA[0], boxB[0])
	interYmin = max(boxA[1], boxB[1])
	interXmax = min(boxA[2], boxB[2])
	interYmax = min(boxA[3], boxB[3])
	
	#Berechnet die Überschneidungsfläche
	interArea = max(0, interXmax - interXmin) * max(0, interYmax - interYmin)
	
	#Berechnet die Fläche der beiden Bounding Boxes
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	
	#Berechnet die größe des Anteils der Überschneidungsfläche an den beiden Bounding Boxes
	ioboxA = interArea / boxAArea
	ioboxB = interArea / boxBArea
	
	#Gibt den kleineren Anteil zurück
	minInter = min(ioboxA, ioboxB)
	
	#Gibt True an, wenn eines der beiden Bounding Boxes zu über 80% in der anderen liegt,
	#der Wert bei keiner der beiden jedoch unter 50% ist
	return (ioboxA > 0.8 or ioboxB > 0.8) and minInter > 0.5
