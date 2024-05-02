import numpy as np
import cv2

# Getting only rectangle contours
def rectContours(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True)
            if len(approx)==4:
                rectCon.append(i)

    rectCon = sorted(rectCon, key = cv2.contourArea, reverse=True)

    return rectCon

def getCornerPoints(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def splitBoxes(img,flag):
    num_of_vsplits=1
    num_of_hsplits=1
    if flag==0:
        num_of_vsplits=28
        num_of_hsplits=15
    elif flag==1:
        num_of_vsplits=28
        num_of_hsplits=12
    elif flag==2:
        num_of_vsplits = 12
        num_of_hsplits = 5
    elif flag==3:
        num_of_vsplits = 3
        num_of_hsplits = 5
    elif flag == 4:
        num_of_vsplits = 20
        num_of_hsplits = 34
    rows = np.vsplit(img,num_of_vsplits)
    boxes = []
    for r in rows:
       cols = np.hsplit(r,num_of_hsplits)
       for box in cols:
           boxes.append(box)
       #cv2.imshow('test',boxes[0])

    return boxes

def splitTestBox(img):
    questions_list = []
    questions = []
    num_hsplit = 5
    num_vsplit = 20
    columns = np.hsplit(img,num_hsplit)
    for i in columns:
        question = np.vsplit(i,num_vsplit)
        for q in question:
            questions_list.append(q)

    for r in questions_list:
        r = cv2.resize(r,(126,31))
        question = np.hsplit(r,7)
        for a in question:
            questions.append(a)

    # for i in range(7):
    #     cv2.imshow(f'test{i}',questions[i])
    #print(questions[0])


    #print(len(questions))
    return questions

    # print(len(questions_list))