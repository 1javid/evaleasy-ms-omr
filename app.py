import cv2
import numpy as np
import utils
from flask import Flask, request, jsonify
from py_eureka_client import eureka_client

app = Flask(__name__)

def register_with_eureka():
    try:
        print("Attempting to register with Eureka.")
        eureka_client.init(eureka_server="http://ms-service-reg:8761/eureka/",
                                           app_name="ms-omr",
                                           instance_port=5000)
        print("Registered with Eureka successfully.")
    except Exception as e:
        print(f"Error when trying to register with Eureka: {e}")

@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    ########
    path = 'images/test7_-_E.jpg'
    alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    digits = ['0','1','2','3','4','5','6','7','8','9']
    answers = ['A','B','C','D','E']
    ###########

    widthImg = 700
    heightImg = 700

    #PREPROCESSING
    #img = cv2.imread(path)
    img = cv2.resize(img, (1100, 1100))

    imgContour = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    imgCanny = cv2.Canny(imgBlur, 10, 70)
    cv2.imshow('Original', imgCanny)

    # FINDING CONTOURS
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContour, contours, -1, (0, 255, 0), 2)

    cv2.imshow('Image with contours', imgContour)
    # FINDING RECTANGLES
    rectCon = utils.rectContours(contours)

    testBox = utils.getCornerPoints(rectCon[0])
    lastNameBox = utils.getCornerPoints(rectCon[1])
    nameBox = utils.getCornerPoints(rectCon[2])
    assesmentIDBox = utils.getCornerPoints(rectCon[3])
    studentIDBox = utils.getCornerPoints(rectCon[4])
    variantBox = utils.getCornerPoints(rectCon[7])

    #### reordering corner points of rectanle
    testBox = utils.reorder(testBox)
    lastNameBox = utils.reorder(lastNameBox)
    nameBox = utils.reorder(nameBox)
    assesmentIDBox = utils.reorder(assesmentIDBox)
    studentIDBox = utils.reorder(studentIDBox)
    variantBox = utils.reorder(variantBox)

    ###   transforming image to bird-eye view

    ### TEST BOX
    ptT1 = np.float32(testBox)
    ptT2 = np.float32([[0, 0], [615, 0], [0, 620], [615, 620]])
    matrixT = cv2.getPerspectiveTransform(ptT1, ptT2)
    imgWarpColoredT = cv2.warpPerspective(img, matrixT, (615, 620))

    ##LAST NAME BOX
    ptL1 = np.float32(lastNameBox)
    ptL2 = np.float32([[0, 0], [615, 0], [0, 616], [615, 616]])
    matrixL = cv2.getPerspectiveTransform(ptL1, ptL2)
    imgWarpColoredL = cv2.warpPerspective(img, matrixL, (615, 616))

    ## NAME BOX
    ptN1 = np.float32(nameBox)
    ptN2 = np.float32([[0, 0], [624, 0], [0, 616], [624, 616]])
    matrixN = cv2.getPerspectiveTransform(ptN1, ptN2)
    imgWarpColoredN = cv2.warpPerspective(img, matrixN, (624, 616))

    # ASSESMENT ID BOX
    ptA1 = np.float32(assesmentIDBox)
    ptA2 = np.float32([[0, 0], [615, 0], [0, 624], [615, 624]])
    matrixA = cv2.getPerspectiveTransform(ptA1, ptA2)
    imgWarpColoredA = cv2.warpPerspective(img, matrixA, (615, 624))

    # STUDENT ID BOX
    ptS1 = np.float32(studentIDBox)
    ptS2 = np.float32([[0, 0], [615, 0], [0, 624], [615, 624]])
    matrixS = cv2.getPerspectiveTransform(ptS1, ptS2)
    imgWarpColoredS = cv2.warpPerspective(img, matrixS, (615, 624))

    # VARIANT BOX
    ptV1 = np.float32(variantBox)
    ptV2 = np.float32([[0, 0], [615, 0], [0, 624], [615, 624]])
    matrixV = cv2.getPerspectiveTransform(ptV1, ptV2)
    imgWarpColoredV = cv2.warpPerspective(img, matrixV, (615, 624))

    # Apply threshold to get marked answers
    # TEST BOX
    imgWarpGrayT = cv2.cvtColor(imgWarpColoredT, cv2.COLOR_BGR2GRAY)

    imgThreshT = cv2.threshold(imgWarpGrayT, 200, 255, cv2.THRESH_BINARY_INV)[1]

    # imgThreshT = cv2.resize(imgThreshT,(622,620))
    # imgThreshS = cv2.resize(imgThreshS,(350,348))
    # LAST NAME BOX
    imgWarpGrayL = cv2.cvtColor(imgWarpColoredL, cv2.COLOR_BGR2GRAY)
    imgThreshL = cv2.threshold(imgWarpGrayL, 200, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('Warp Gray Img', imgThreshL)
    ######NAME BOX######
    imgWarpGrayN = cv2.cvtColor(imgWarpColoredN, cv2.COLOR_BGR2GRAY)
    imgThreshN = cv2.threshold(imgWarpGrayN, 180, 255, cv2.THRESH_BINARY_INV)[1]

    # ASSESMENT ID BOX
    imgWarpGrayA = cv2.cvtColor(imgWarpColoredA, cv2.COLOR_BGR2GRAY)
    imgThreshA = cv2.threshold(imgWarpGrayA, 220, 255, cv2.THRESH_BINARY_INV)[1]
    imgThreshA = cv2.resize(imgThreshA, (350, 348))
    # STUDENT ID BOX
    imgWarpGrayS = cv2.cvtColor(imgWarpColoredS, cv2.COLOR_BGR2GRAY)
    imgThreshS = cv2.threshold(imgWarpGrayS, 220, 255, cv2.THRESH_BINARY_INV)[1]
    imgThreshS = cv2.resize(imgThreshS, (350, 348))
    # VARIANT BOX
    imgWarpGrayV = cv2.cvtColor(imgWarpColoredV, cv2.COLOR_BGR2GRAY)
    imgThreshV = cv2.threshold(imgWarpGrayV, 190, 255, cv2.THRESH_BINARY_INV)[1]
    imgThreshV = cv2.resize(imgThreshV, (200, 201))

    # SPLIT INTO BOXES
    ##### TEST BOX
    boxesT = utils.splitTestBox(imgThreshT)

    # LAST NAME BOX
    boxesL = utils.splitBoxes(imgThreshL, 0)

    #####NAME BOX#####
    boxesN = utils.splitBoxes(imgThreshN, 1)

    # ASSESMENT ID BOX
    boxesA = utils.splitBoxes(imgThreshA, 2)
    # for i in range(31,len(boxesA)):
    #   boxesA[i] = cv2.resize(boxesA[i],(100,100))
    # cv2.imshow(f"Test{i}",boxes[i])

    # STUDENT ID BOX
    boxesS = utils.splitBoxes(imgThreshS, 2)
    # for i in range(31, len(boxesS)):
    #     boxesS[i] = cv2.resize(boxesS[i], (100, 100))
    # cv2.imshow(f"Test{i}", boxes[i])

    # VARIANT BOX
    boxesV = utils.splitBoxes(imgThreshV, 3)
    # for i in range(10, len(boxesV)):
    #     boxesV[i] = cv2.resize(boxesV[i], (100, 100))
    #     cv2.imshow(f"Test{i}", boxes[i])

    ###CALCULATING NON ZERO PIXEL VALUES

    ###########################TEST BOX ##########################

    # for i in range(30):
    #     cv2.imshow(f'{i+19}',boxesT[i])

    ##################### TEST BOX ######################

    myPixelArrT = np.zeros((100, 5))
    countTC = 0
    countTR = 0

    for i in range(5):
        j = i * 20
        for j in range(j, j + 20):
            countTC = 0
            for k in range(1, 6):
                totalPixels = cv2.countNonZero(boxesT[j * 7 + k])
                myPixelArrT[countTR][countTC] = totalPixels
                countTC = countTC + 1
            countTR = countTR + 1

    myIndexT = []

    max_indices = np.argmax(myPixelArrT, axis=1)

    answersT = {}

    for i, sublist in enumerate(myPixelArrT):
        indices = [answers[j] for j, value in enumerate(sublist) if value > 300]
        if indices:
            answersT[i + 1] = indices

    #
    # for
    #
    # for r in range()

    ########################### LAST NAME ################################

    myPixelArrL = np.zeros((26, 15))
    countLC = 0
    countLR = 0

    for r in range(30, 45):
        for i in range(26):
            totalPixels = cv2.countNonZero(boxesL[r + 15 * i])
            myPixelArrL[countLR][countLC] = totalPixels
            countLR += 1
            if countLR == 26:
                countLC += 1
                countLR = 0
    myPixelArrL = myPixelArrL.transpose()

    myIndexL = []

    for x in range(15):
        arr = myPixelArrL[x]
        if x == 0 or x == 14:
            if np.amax(arr) > 540:
                myIndexVal = np.where(arr == np.amax(arr))
                myIndexL.append(myIndexVal[0][0])
            else:
                myIndexL.append(' ')
        elif np.amax(arr) > 500:
            myIndexVal = np.where(arr == np.amax(arr))
            myIndexL.append(myIndexVal[0][0])
        else:
            myIndexL.append(' ')

    lastName = ""
    while myIndexL and myIndexL[-1] == ' ':
        myIndexL.pop()

    for i in range(len(myIndexL)):
        if myIndexL[i] == " ":
            lastName = lastName + ' '
        else:
            lastName = lastName + alphabet[myIndexL[i]]

    # print(myPixelArrL)

    ####################### NAME #########################
    myPixelArrN = np.zeros((26, 12))
    countNC = 0
    countNR = 0
    for r in range(24, 36):
        for i in range(26):
            totalPixels = cv2.countNonZero(boxesN[r + 12 * i])
            myPixelArrN[countNR][countNC] = totalPixels
            countNR += 1
            if countNR == 26:
                countNC += 1
                countNR = 0
    myPixelArrN = myPixelArrN.transpose()

    myIndexN = []
    for x in range(12):
        arr = myPixelArrN[x]
        if x == 0 or x == 11:
            if np.amax(arr) > 650:
                myIndexVal = np.where(arr == np.amax(arr))
                myIndexN.append(myIndexVal[0][0])
            else:
                myIndexN.append(' ')
        elif np.amax(arr) > 550:
            myIndexVal = np.where(arr == np.amax(arr))
            myIndexN.append(myIndexVal[0][0])
        else:
            myIndexN.append(' ')

    name = ""
    while myIndexN and myIndexN[-1] == ' ':
        myIndexN.pop()

    for i in range(len(myIndexN)):
        if myIndexN[i] == ' ':
            name = name + ' '
        else:
            name = name + alphabet[myIndexN[i]]

    ##################### ASSESMENT ID ###############################

    myPixelArrA = np.zeros((10, 5))
    countAC = 0
    countAR = 0
    for r in range(10, 15):
        for i in range(10):
            totalPixels = cv2.countNonZero(boxesA[r + 5 * i])
            myPixelArrA[countAR][countAC] = totalPixels
            countAR += 1
            if countAR == 10:
                countAC += 1
                countAR = 0

    myPixelArrA = myPixelArrA.transpose()
    # FINDING INDEX VALUE OF CHOSEN ANSWERS
    myIndexA = []
    for x in range(5):
        arr = myPixelArrA[x]

        if x == 0 or x == 4:
            if np.amax(arr) > 1470:
                myIndexVal = np.where(arr == np.amax(arr))
                myIndexA.append(myIndexVal[0][0])
            else:
                myIndexA.append(' ')
        elif np.amax(arr) > 1270:
            myIndexVal = np.where(arr == np.amax(arr))
            myIndexA.append(myIndexVal[0][0])
        else:
            myIndexA.append(' ')

    assesmentID = ""
    while myIndexA and myIndexA[-1] == ' ':
        myIndexA.pop()

    for i in range(len(myIndexA)):
        if myIndexA[i] == ' ':
            assesmentID = assesmentID + ' '
        else:
            assesmentID = assesmentID + digits[myIndexA[i]]

    ##################### STUDENT ID ###############################

    myPixelArrS = np.zeros((10, 5))
    countSC = 0
    countSR = 0
    for r in range(10, 15):
        for i in range(10):
            totalPixels = cv2.countNonZero(boxesS[r + 5 * i])
            myPixelArrS[countSR][countSC] = totalPixels
            countSR += 1
            if countSR == 10:
                countSC += 1
                countSR = 0

    myPixelArrS = myPixelArrS.transpose()
    # FINDING INDEX VALUE OF CHOSEN ANSWERS
    myIndexS = []
    for x in range(5):
        arr = myPixelArrS[x]
        if x == 4 or x == 0:
            if np.amax(arr) > 1490:
                myIndexVal = np.where(arr == np.amax(arr))
                myIndexS.append(myIndexVal[0][0])
            else:
                myIndexS.append(' ')
        elif np.amax(arr) > 1300:
            myIndexVal = np.where(arr == np.amax(arr))
            myIndexS.append(myIndexVal[0][0])
        else:
            myIndexS.append(' ')

    studentID = ""
    while myIndexS and myIndexS[-1] == ' ':
        myIndexS.pop()

    for i in range(len(myIndexS)):
        if myIndexS[i] == ' ':
            studentID = studentID + ' '
        else:
            studentID = studentID + digits[myIndexS[i]]

    ##################### VARIANT ###############################
    myPixelArrV = np.zeros(5)

    for i in range(5):
        totalPixels = cv2.countNonZero(boxesV[i + 10])
        myPixelArrV[i] = totalPixels

    max_value = np.max(myPixelArrV)

    myIndexV = []
    for x in range(5):
        if myPixelArrV[x] > 1700:
            myIndexV.append(x)

    variant = ''
    for i in range(len(myIndexV)):
        variant = variant + alphabet[myIndexV[i]]

    # DISPLAYING

    print(lastName)
    print(name)
    print(assesmentID)
    print(studentID)
    print(variant)
    print(answersT)

    #
    results = {
        'lastName': lastName,
        'name': name,
        'assessmentID': assesmentID,
        'studentID': studentID,
        'variant': variant,
        'answers': answersT
    }

    # Send results in JSON format
    return jsonify(results)

if __name__ == '__main__':
    print("Starting application.")
    register_with_eureka()
    app.run(host='0.0.0.0', port=5000)
