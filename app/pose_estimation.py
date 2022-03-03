import cv2 as cv
import numpy as np
from pathlib import Path
import imutils

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
# file = BASE_DIR / "examples" / "test1.mp4"

# pre-trained caffe model files
protoFile = str(MODEL_DIR / "pose_deploy_linevec.prototxt")
#protoFile = str(MODEL_DIR / "pose_deploy_linevec_faster_4_stages.prototxt")
weightsFile = str(MODEL_DIR / "pose_iter_440000.caffemodel")
# to check if the device has gpu
gpu = False
if cv.cuda.getCudaEnabledDeviceCount() > 0:
    gpu = True
inHeight = 368
threshold = 0.2

'''
COCO Output Format
BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
              "LEye": 15, "REar": 16, "LEar": 17}

'''
nPoints = 18
# pairs of body parts; eg. "Neck" and "RShoulder", "Neck", "LShoulder", etc
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
              [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
              [2, 17], [5, 16]]


# get keypoints using Non Maximum Suppression (to get rid of undetected frames)
# reference from https://github.com/spmallick/learnopencv/blob/master/OpenPose-Multi-Person/multi-person-openpose.py
def getKeypoints(probMap, threshold=0.1):
    mapSmooth = cv.GaussianBlur(probMap, (3, 3), 0, 0)
    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []
    # find the blobs
    contours, _ = cv.findContours(mapMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints

def get_min_max(val, min_val, max_val):
    if val < min_val:
        min_val = val
    if val > max_val:
        max_val = val
    return min_val, max_val

def run_pose_estimation(frame, frame1, net):
    # get difference between two frames
    diff = cv.absdiff(frame, frame1)
    gray = cv.cvtColor(diff, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 3), 0)
    _, thresh = cv.threshold(blur, 25, 255, cv.THRESH_BINARY)
    dilated = cv.dilate(thresh, None, iterations=2)
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # only run the keypoints detection when there is motion in the frame
    # in order to prevent getting false positive
    for c in contours:
        if cv.contourArea(c) > 5000:
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            inWidth = int((inHeight / frameHeight) * frameWidth)
            inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                            (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inpBlob)
            output = net.forward()

            detected_keypoints = []
            keypoints_list = np.zeros((0, 3))
            keypoint_id = 0

            for part in range(nPoints):
                probMap = output[0, part, :, :]
                probMap = cv.resize(probMap, (frameWidth, frameHeight))
                keypoints = getKeypoints(probMap, threshold)
                keypoints_with_id = []
                for i in range(len(keypoints)):
                    keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                    keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                    keypoint_id += 1

                detected_keypoints.append(keypoints_with_id)

            # to get min max values of keypoints
            min_x = float("inf")
            max_x = float("-inf")
            min_y = float("inf")
            max_y = float("-inf")
            for i in range(nPoints):
                if len(detected_keypoints[i]) != 0:
                    x, y = detected_keypoints[i][0][0:2]
                    min_x, max_x = get_min_max(x, min_x, max_x)
                    min_y, max_y = get_min_max(y, min_y, max_y)
                    cv.circle(frame, detected_keypoints[i][0][0:2], 3, [0, 255, 0], -1, cv.LINE_AA)
            diff_x = max_x - min_x
            diff_y = max_y - min_y
            if diff_x > diff_y:
                cv.putText(frame, 'Fall Detected!', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                cv.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    #cv.imshow("pose estimation", frame)
    return frame
    

def load_video(video):
    # initialize caffee model weights
    net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
    if gpu: # using gpu
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)  # using CPU
    camera = cv.VideoCapture(video)
    count = 0
    while True:
        success, frame = camera.read()  # read the camera frame
        _, frame1 = camera.read()
        frame = imutils.resize(frame, width=600)
        frame1 = imutils.resize(frame1, width=600)
        if not success:
            camera.release()
            break
        else:
            count += 5
            camera.set(cv.CAP_PROP_POS_FRAMES, count)
            
            # run pose estimation
            frame = run_pose_estimation(frame, frame1, net)
            
            # display the frame on web
            # encode the frame in JPEG format
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # yield the frame in the byte format
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 


def load_webcam():
    camera = cv.VideoCapture(0)
    while True:
        success, frame = camera.read()
        _, frame1 = camera.read()
        if not success:
            camera.release()
            break
        else:
            # get difference between two frames
            diff = cv.absdiff(frame, frame1)
            gray = cv.cvtColor(diff, cv.COLOR_RGB2GRAY)
            blur = cv.GaussianBlur(gray, (5, 3), 0)
            _, thresh = cv.threshold(blur, 25, 255, cv.THRESH_BINARY)
            dilated = cv.dilate(thresh, None, iterations=2)
            contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv.contourArea(c) > 5000:
                    x, y, w, h = cv.boundingRect(c)
                    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # display the frame on web
            # encode the frame in JPEG format
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # yield the frame in the byte format
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
# if __name__ == "__main__":
#     load_video(0)