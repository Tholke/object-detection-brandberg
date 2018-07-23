#Segmentiert Bilder, indem der Watershed-Algorithmus angewendet wird
import cv2
import numpy as np

def computeWatershed(image, saveImagePath):
    #Bild wird geladen und in Graustufen umgewandelt
    img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Bild wird durch die Otsu-Methode in schwarz weiß umgewandelt
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #Weißes Rauschen wird durch morphologisches Opening entfernt
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

    #Sicherer Hintergrund wird gefunden
    sure_bg = cv2.dilate(opening, kernel, iterations = 3)

    #Sicherer Vordergrund wird durch distance Transform und threshholding gefunden
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    #Unsichere Regionen
    unknown = cv2.subtract(sure_bg, sure_fg)

    #Marker werden erstellt
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1

    #Marker werden auf schwarz gesetzt. Und der watershed-Algorithmus darauf angewendet
    markers[unknown==255] = 0
    markers = cv2.watershed(img, markers)

    height, width, channels = img.shape

    #Ein weißes Bild, mit den Maßen des Originalbildes, wird erstellt und bei jedem Marker wird das Pixel schwarz gefärbt
    markerImg = np.ones((height, width, channels), np.uint8) * 255
    markerImg[markers == -1] = [0, 0, 0]

    #Das Bild wird auf 500 Pixel höhe skaliert, das Seitenverhältnis bleibt erhalten. (Nur zu Anzeigezwecken)
    rs_height = 500
    rs_width = int(height * rs_height / width)
    resizedImg = cv2.resize(markerImg, (rs_height, rs_width))
    img = cv2.resize(img, (rs_height, rs_width))

    #Das Originalbild und das Markerbild werden angezeigt
    cv2.imshow("imgwithborders",img)
    cv2.imshow("markers",resizedImg)
    
    #Wird die Taste 's' gedrückt, wird das Bild abgespeichert
    k = cv2.waitKey(0) & 0xFF
    if k == ord('s'):
        cv2.imwrite(saveImagePath, markerImg)
        print('Bild in Pfad:', saveImagePath, 'gespeichert')
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    computeWatershed(image)