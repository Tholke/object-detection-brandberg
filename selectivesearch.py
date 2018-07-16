import cv2
import numpy as np

def main():
    img = cv2.imread('/Users/thomas/Desktop/Uni/SoSe18/KI/BOOK-ZID0824731/images-raw/BOOK-0824731-0056.jpg')
    img2 = img.copy()
    
    newWidth = int(img.shape[1]*600/img.shape[0])
    img = cv2.resize(img, (newWidth, 600))
    
    selectiveSearch = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selectiveSearch.setBaseImage(img)
    selectiveSearch.switchToSelectiveSearchFast()

    rects = selectiveSearch.process()
    print('Total Number of Region Proposals:',len(rects))

    rectsToDelete = []
    #print(rects[0])
    
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        area = w * h
        
        if area > 90000 or area < 100 or w < 50 or h < 50:
            rectsToDelete.append(i)
    
    print(len(rects), "-", len(rectsToDelete), "=", len(rects)-len(rectsToDelete))

    for j in reversed(rectsToDelete):
        rects = np.delete(rects, j, 0)
    rects = rects[:500]

    print(len(rects))
    images = []
    hscale = img2.shape[0] / img.shape[0]
    wscale = img2.shape[1] / img.shape[1]
    
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        rectImg1 = img2[int(y*hscale):int((y+h)*hscale), int(x*wscale):int((x+w)*wscale)]
        #rectImg2 = img[y:y+h, x:x+w]
        images.append(rectImg1)
        #images.append(rectImg2)

    cv2.imshow("output", images[0])
    cv2.imwrite('../one.jpg', images[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()