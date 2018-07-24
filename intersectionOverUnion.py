def intersectionOverUnion(boxA, boxB):
    interXmin = max(boxA[0], boxB[0])
    interYmin = max(boxA[1], boxB[1])
    interXmax = min(boxA[2], boxB[2])
    interYmax = min(boxA[3], boxB[3])
  
    interArea = max(0, interXmax - interXmin) * max(0, interYmax - interYmin)
 
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
    ioboxA = interArea / boxAArea
    ioboxB = interArea / boxBArea
    
    minInter = min(ioboxA, ioboxB)
    
    return (ioboxA > 0.8 or ioboxB > 0.8) and minInter > 0.5
    
if __name__ == "__main__":
    intersectionOverUnion(boxA, boxB)