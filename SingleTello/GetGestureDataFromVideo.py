#
# OpenPose used under Copyright with permission
# Copyright (c) 2014-2017 The Regents of the University of California (Regents)
# All rights reserved.
#

from PIL import Image
from PIL import ImageTk
import Tkinter as tki
from Tkinter import Toplevel, Scale
import threading
import datetime
import cv2
import os
import time
import platform
import numpy
import sys
from enum import Enum
from numpy import save
from numpy import load
from numpy import asarray
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

#
# Code obtained from https://github.com/CMU-Perceptual-Computing-Lab/openpose
# Copyright (c) 2014-2017 The Regents of the University of California (Regents)
# All rights reserved.
#

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print ("path is " + dir_path)
    try:
        # Windows Import
        if sys.platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/op/python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/op/Release;' +  dir_path + '/op/bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../op/python/openpose/Release');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            import openpose as op
            #from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

except Exception as e:
    print(e)
    sys.exit(-1)

#
# End of code obtained from https://github.com/CMU-Perceptual-Computing-Lab/openpose
#

# Specify the paths for the 2 files
# protoFile = "models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
# weightsFile = "models/pose/mpi/pose_iter_160000.caffemodel"
# nPoints = 15

#distressGesture = []
currentGesture = []

# # Read the network into Memory
# net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


webcam = cv2.VideoCapture('gestures/startPose.mp4')

params = dict()
params["model_folder"] = "models/"
params["face"] = True
params["face_detector"] = 2

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

poseModel = op.PoseModel.BODY_25

while True:
    # # Capture frame-by-frame
    # wcRet, wcFrame = webcam.read()
    #
    # if wcFrame is not None:
    #     # Process Image
    #     datum = op.Datum()
    #     datum.cvInputData = wcFrame
    #     # Need to emplace and pop to get array for processing
    #     opWrapper.emplaceAndPop([datum])
    #     # newImage = datum.cvOutputData[:, :, :]
    #
    #
    #     currentGesture.append(datum.poseKeypoints)
    #
    #     # Display Image
    #     # print("Body keypoints: \n" + str(datum.poseKeypoints))
    #     cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
    #     # Need this line of code for feed to show
    #     if cv2.waitKey(1) == 27:
    #         break
    # else:
    #     data = asarray(currentGesture)
    #     save('distressGestureArms.npy', data)
    #     print("success")
    #     time.sleep(5)
    #     break


    # ARMS ONLY CODE
    # Capture frame-by-frame
    wcRet, wcFrame = webcam.read()

    if wcFrame is not None:
        # Process Image
        datum = op.Datum()
        datum.cvInputData = wcFrame
        # Need to emplace and pop to get array for processing
        opWrapper.emplaceAndPop([datum])
        # newImage = datum.cvOutputData[:, :, :]

        lWristCoordinates = []
        lElbowCoordinates = []
        lShoulderCoordinates = []
        rWristCoordinates = []
        rElbowCoordinates = []
        rShoulderCoordinates = []

        lWrist = 0  # 7
        lElbow = 0  # 6
        lShoulder = 0  # 5
        rWrist = 0  # 4
        rElbow = 0  # 3
        rShoulder = 0  # 2

        for n in op.getPosePartPairs(poseModel):
            if n == 8:
                hip = n
                break
        for n in op.getPosePartPairs(poseModel):
            if n == 7:
                lWrist = n
                break
        for n in op.getPosePartPairs(poseModel):
            if n == 6:
                lElbow = n
                break
        for n in op.getPosePartPairs(poseModel):
            if n == 5:
                lShoulder = n
                break
        for n in op.getPosePartPairs(poseModel):
            if n == 4:
                rWrist = n
                break
        for n in op.getPosePartPairs(poseModel):
            if n == 3:
                rElbow = n
                break
        for n in op.getPosePartPairs(poseModel):
            if n == 2:
                rShoulder = n
                break
        for n in op.getPosePartPairs(poseModel):
            if n == 1:
                neck = n
                break


        lWristCoordinates = datum.poseKeypoints[0, lWrist, 0:2]
        lElbowCoordinates = datum.poseKeypoints[0, lElbow, 0:2]
        lShoulderCoordinates = datum.poseKeypoints[0, lShoulder, 0:2]
        rWristCoordinates = datum.poseKeypoints[0, rWrist, 0:2]
        rElbowCoordinates = datum.poseKeypoints[0, rElbow, 0:2]
        rShoulderCoordinates = datum.poseKeypoints[0, rShoulder, 0:2]

        currentGesture.append(lWristCoordinates)
        currentGesture.append(lElbowCoordinates)
        currentGesture.append(lShoulderCoordinates)
        currentGesture.append(rWristCoordinates)
        currentGesture.append(rElbowCoordinates)
        currentGesture.append(rShoulderCoordinates)

        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        # Need this line of code for feed to show
        if cv2.waitKey(1) == 27:
            break
    else:
        data = asarray(currentGesture)
        save('distressGestureArms.npy', data)
        print("success")
        time.sleep(5)
        break