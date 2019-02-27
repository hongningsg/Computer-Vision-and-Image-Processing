import cv2
import time
import numpy as np
import sys
import math
import numba
from random import randint

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
matching_size = 13
last_round = 0.5

partsMap = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']

connectionPair = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]


PAFs = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
          [47,48], [49,50], [53,54], [51,52], [55,56],
          [37,38], [45,46]]

colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]


def getKeypoints(probMap, threshold=0.1):

    imgBlur = cv2.GaussianBlur(probMap,(3,3),0,0)

    imgThreshold = np.uint8(imgBlur>threshold)
    keypoints = []


    _, VectorFields, _ = cv2.findVectorFields(imgThreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for vf in VectorFields:
        blobMask = np.zeros(imgThreshold.shape)
        blobMask = cv2.fillConvexPoly(blobMask, vf, 1)
        maskedProbMap = imgBlur * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints



@numba.jit
def getValidPairs(output):
    valid_pairs = []
    invalid_pairs = []
    nsample = 10
    pairScore = 0.1
    confidenceScore = 0.7

    for k in range(len(PAFs)):

        pafA = output[0, PAFs[k][0], :, :]
        pafB = output[0, PAFs[k][1], :, :]
        pafA = cv2.resize(pafA, (frameWidth, frameHeight))
        pafB = cv2.resize(pafB, (frameWidth, frameHeight))


        candA = detected_keypoints[connectionPair[k][0]]
        candB = detected_keypoints[connectionPair[k][1]]
        nA = len(candA)
        nB = len(candB)


        if( nA != 0 and nB != 0):
            valid_pair = np.zeros((0,3))
            for i in range(nA):
                max_j=-1
                maxScore = -1
                found = 0
                for j in range(nB):
                    # Find d_ij
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue

                    coordinate_int = list(zip(np.linspace(candA[i][0], candB[j][0], num=nsample),
                                            np.linspace(candA[i][1], candB[j][1], num=nsample)))
                    paf_interp = []
                    for k in range(len(coordinate_int)):
                        paf_interp.append([pafA[int(round(coordinate_int[k][1])), int(round(coordinate_int[k][0]))],
                                           pafB[int(round(coordinate_int[k][1])), int(round(coordinate_int[k][0]))] ])

                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores)/len(paf_scores)


                    if ( len(np.where(paf_scores > pairScore)[0]) / nsample ) > confidenceScore :
                        if avg_paf_score > maxScore:
                            max_j = j
                            maxScore = avg_paf_score
                            found = 1

                if found:
                    valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)


            valid_pairs.append(valid_pair)
        else: 
            invalid_pairs.append(k)
            valid_pairs.append([])
    return valid_pairs, invalid_pairs




@numba.jit
def getKeypoints_Person(valid_pairs, invalid_pairs):

    Keypoints_Person = -1 * np.ones((0, 19))

    for k in range(len(PAFs)):
        if k not in invalid_pairs:
            partAs = valid_pairs[k][:,0]
            partBs = valid_pairs[k][:,1]
            indexA, indexB = np.array(connectionPair[k])

            for i in range(len(valid_pairs[k])):
                found = 0
                pindex = -1
                for j in range(len(Keypoints_Person)):
                    if Keypoints_Person[j][indexA] == partAs[i]:
                        pindex = j
                        found = 1
                        break

                if found:
                    Keypoints_Person[pindex][indexB] = partBs[i]
                    Keypoints_Person[pindex][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]


                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]

                    row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
                    Keypoints_Person = np.vstack([Keypoints_Person, row])
    return Keypoints_Person

def readModelPose(fileName):
    img = cv2.imread(fileName)
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    frameWidth = img.shape[1]
    frameHeight = img.shape[0]
    inHeight = 368
    inWidth = int((inHeight / frameHeight) * frameWidth)
    inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    detected_keypoints = []
    keypoints_list = np.zeros((0, 3))
    keypoint_id = 0
    threshold = 0.1
    for part in range(nPoints):
        probMap = output[0, part, :, :]
        probMap = cv2.resize(probMap, (img.shape[1], img.shape[0]))
        keypoints = getKeypoints(probMap, threshold)
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)

    frameClone = img.copy()
    valid_pairs, invalid_pairs = getValidPairs(output)
    Keypoints_Person = getKeypoints_Person(valid_pairs, invalid_pairs)

    for i in range(17):
        for n in range(len(Keypoints_Person)):
            index = Keypoints_Person[n][np.array(connectionPair[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
    cv2.imshow("Model Pose", frameClone)
    cv2.waitKey(0)
    return getVectors(Keypoints_Person, detected_keypoints)


def Normal(A, B):
    return math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)

@numba.jit
def getSimilarity(vector1, vector2):
    similarity = 0
    num = 0
    for i in range(matching_size):
        if -1 in vector1[i] or -1 in vector2[i]:
            continue
        else:
            num += 1
            normA = Normal(vector1[i][1], vector1[i][0])
            A = ((vector1[i][1][0] - vector1[i][0][0])/normA, (vector1[i][1][1] - vector1[i][0][1])/normA)
            normB = Normal(vector2[i][1], vector2[i][0])
            B = ((vector2[i][1][0] - vector2[i][0][0])/normB, (vector2[i][1][1] - vector2[i][0][1])/normB)
            similarity += (A[0]*B[0] + A[1]*B[1])/(math.sqrt(A[0]**2 + A[1]**2) * math.sqrt(B[0]**2 + B[1]**2))
    if num == 0:
        return 0
    return similarity/num

def findIndex(Alist, index):
    for part in Alist:
        if index == part[3]:
            return part[0], part[1]
    return -1, -1

def getVectors(Person, Keypoints):
    connect = [(0,1), (1,5), (5,6), (6,7), (1,2), (2,3), (3,4), (1,11), (11,12), (12,13), (1,8), (8,9), (9,10)]
    personVectors=[]
    for p in Person:
        pointsVectors = []
        for pair in connect:
            Pvector = []
            for i in range(2):
                if p[pair[i]] == -1:
                    A = -1
                else:
                    personIndex = int(p[pair[i]])
                    x, y = findIndex(Keypoints[pair[i]], personIndex)
                    A = (x, y)
                Pvector.append(A)
            pointsVectors.append(Pvector)
        personVectors.append(pointsVectors)
    return personVectors

# read Model
img = cv2.imread(sys.argv[1])
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
frameWidth = img.shape[1]
frameHeight = img.shape[0]
inHeight = 368
inWidth = int((inHeight / frameHeight) * frameWidth)
inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob)
output = net.forward()
detected_keypoints = []
keypoints_list = np.zeros((0, 3))
keypoint_id = 0
threshold = 0.1
for part in range(nPoints):
    probMap = output[0, part, :, :]
    probMap = cv2.resize(probMap, (img.shape[1], img.shape[0]))
    keypoints = getKeypoints(probMap, threshold)
    keypoints_with_id = []
    for i in range(len(keypoints)):
        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
        keypoints_list = np.vstack([keypoints_list, keypoints[i]])
        keypoint_id += 1

    detected_keypoints.append(keypoints_with_id)

frameClone = img.copy()
valid_pairs, invalid_pairs = getValidPairs(output)
Keypoints_Person = getKeypoints_Person(valid_pairs, invalid_pairs)

for i in range(17):
    for n in range(len(Keypoints_Person)):
        index = Keypoints_Person[n][np.array(connectionPair[i])]
        if -1 in index:
            continue
        B = np.int32(keypoints_list[index.astype(int), 0])
        A = np.int32(keypoints_list[index.astype(int), 1])
        cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
cv2.imshow("Model Pose", frameClone)
cv2.waitKey(0)
modelVector = getVectors(Keypoints_Person, detected_keypoints)[0]

# modelVector = readModelPose("SampleModel.png")
# print(modelVector)
video = cv2.VideoCapture(sys.argv[2])
still, image1 = video.read()
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
frameWidth = image1.shape[1]
frameHeight = image1.shape[0]
# frameWidth = image1.shape[1]/10
# frameHeight = image1.shape[0]/10
inHeight = 368
# inHeight = 368
inWidth = int((inHeight/frameHeight)*frameWidth)
vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (image1.shape[1],image1.shape[0]))
frame_num = 1
while video.isOpened():
    t = time.time()



    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()


    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1

    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = getKeypoints(probMap, threshold)

        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)


    frameClone = image1.copy()

    valid_pairs, invalid_pairs = getValidPairs(output)
    Keypoints_Person = getKeypoints_Person(valid_pairs, invalid_pairs)

    for i in range(17):
        for n in range(len(Keypoints_Person)):
            index = Keypoints_Person[n][np.array(connectionPair[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

    k = cv2.waitKey(1)
    if k == 27:
        sys.exit(1)
    currVector = getVectors(Keypoints_Person, detected_keypoints)

    for pvec in currVector:
        similarity = getSimilarity(pvec, modelVector)
        if similarity > 0.9:
            nose = pvec[0][0]
            neck = pvec[0][1]
            if type(nose) == int or type(neck) == int:
                continue
            length = math.sqrt(2) * math.sqrt((nose[1] - neck[1]) ** 2 + (nose[0] - neck[0]) ** 2)
            pt1 = (int(nose[0] - length), int(nose[1] - length))
            pt2 = (int(nose[0] + length), int(nose[1] + length))
            cv2.rectangle(frameClone, pt1, pt2, (0, 0, 100), 3)
            accuracy = similarity * 100
            cv2.putText(frameClone, "Match - Acc:" + str(int(accuracy)) + "%", (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100), 2)
    vid_writer.write(frameClone)
    print("Processed frame", frame_num, ", time use:", time.time() - t)
    frame_num += 1
    still, image1 = video.read()
    if not still:
        break
video.release()
vid_writer.release()
cv2.destroyAllWindows()
