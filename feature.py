import cv2
import numpy as np
import os

path =r'C:\Users\shubh\OneDrive\Desktop\feature\images'
orb = cv2.ORB_create(nfeatures=1000)
images = []
classname = []
myList = os.listdir(path)

for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classname.append(os.path.splitext(cl)[0])
print(classname)




def findDes(images):
    deslist = []

    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        deslist.append(des)
    return deslist
    
deslist = findDes(images)
print(len(deslist))


def findId(img, deslist, thres = 12):
    kp2,des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalValue = -1
    try:
        for des in deslist:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    print(matchList)
    if len(matchList) !=0:
        if max(matchList) > thres:
            finalValue = matchList.index(max(matchList))
    return finalValue




cap = cv2.VideoCapture(0)
while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    id = findId(img2, deslist)
    if id != -1:
        cv2.putText(imgOriginal, classname[id], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv2.imshow('img2', imgOriginal)
    cv2.waitKey(1)
