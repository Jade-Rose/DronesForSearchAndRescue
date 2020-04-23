#
# Code based on sample code provided by Ryze via Official Github repository under MIT License
# Copyright (c) 2018 Ryze
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
from targetInDistress import TargetInDistess


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

params = dict()
params["model_folder"] = "models/"
params["face"] = True
params["face_detector"] = 2

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()


class TelloUI:
    """Wrapper class to enable the GUI."""

    target = TargetInDistess()

    # Nested class used to enumerate drone states
    class DroneState(Enum):
        idle = 0
        searching = 1
        focusingBody = 2
        approaching = 3

    def __init__(self,tello,outputpath):
        """
        Initial all the element of the GUI,support by Tkinter

        :param tello: class interacts with the Tello drone.

        Raises:
            RuntimeError: If the Tello rejects the attempt to enter command mode.
        """

        self.img2 = None

        # Specify the paths for the 2 files
        self.protoFile = "models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
        self.weightsFile = "models/pose/mpi/pose_iter_160000.caffemodel"
        self.nPoints = 15

        # Variables store the dimensions of the drone frame
        self.frameWidth = 960
        self.frameHeight = 690

        self.distressGesture = load('distressGestureArms.npy')
        self.distressTriggerPose = load('distressTriggerPose.npy')
        self.currentGesture = [] # Array for storing model gesture data

        # Arrays for storing live gesture data
        self.currentFallenGesture = []
        self.confirmFallenGesture = []

        self.recording = False
        self.recordingFallen = False
        self.waveDetected = False
        self.fallenDetected = False

        self.state = self.DroneState.searching

        self.idOfPersonInDistress = None

        # Read the network into Memory
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)

        # Import Haar cascades
        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.profileFaceCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
        self.bodyCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

        self.faces = [] # Array for storing detected face
        self.profileFaces = [] # Array for storing faces detected from the left side
        self.altProfileFaces = []  # Array for storing faces detected from the left side

        #TELLO STREAM
        self.tello = tello # videostream device
        self.outputPath = outputpath # the path that save pictures created by clicking the takeSnapshot button 
        self.frame = None  # frame read from h264decoder and used for pose recognition 
        self.thread = None # thread of the Tkinter mainloop
        self.stopEvent = None  

        self.cameraImage = None

        # control variables
        self.distance = 0.1  # default distance for 'move' cmd
        self.degree = 30  # default degree for 'cw' or 'ccw' cmd

        # Initial search paramaters
        self.searchLoops = 5 # how many times to loop around in search pattern
        self.searchDistance = 20 # how far to travel along edge of expanding square
        self.edgesTravelled = 0

        # if the flag is TRUE, the auto-takeoff thread will stop waiting for the response from tello
        self.quit_waiting_flag = False
        
        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        # create buttons
        self.btn_snapshot = tki.Button(self.root, text="Snapshot!",
                                       command=self.takeSnapshot)
        self.btn_snapshot.pack(side="bottom", fill="both",
                               expand="yes", padx=10, pady=5)

        self.btn_pause = tki.Button(self.root, text="Pause", relief="raised", command=self.pauseVideo)
        self.btn_pause.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_landing = tki.Button(
            self.root, text="Open Command Panel", relief="raised", command=self.openCmdWindow)
        self.btn_landing.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)
        
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("TELLO Controller")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        # the sending_command will send command to tello every 5 seconds
        self.sending_command_thread = threading.Thread(target = self._sendingCommand)

    def videoLoop(self):
        """
        The mainloop thread of Tkinter 
        Raises:
            RuntimeError: To get around a RunTime error that Tkinter throws due to threading.
        """
        try:
            # start the thread that get GUI image and draw skeleton
            time.sleep(0.5)
            #self.sending_command_thread.start()


            # Take off when launched
            self.tello.takeoff()
            time.sleep(5)
            batteryLevel = self.tello.send_command('battery?')
            print(batteryLevel)

            while not self.stopEvent.is_set():
                system = platform.system()

            # read the frame for GUI show
                self.frame = self.tello.read()
                if self.frame is None or self.frame.size == 0:
                    continue 
            
            # transfer the format from frame to image         
                image = Image.fromarray(self.frame)

                open_cv_image = numpy.array(image)
                # Convert RGB to BGR
                self.cameraImage = open_cv_image[:, :, ::-1].copy()


            # we found compatibility problem between Tkinter,PIL and Macos,and it will 
            # sometimes result the very long preriod of the "ImageTk.PhotoImage" function,
            # so for Macos,we start a new thread to execute the _updateGUIImage function.
                if system =="Windows" or system =="Linux":
                    self._updateGUIImage(image)

                    # System behaviour determined by state
                    if self.state == self.DroneState.searching:
                        self.Search(self.cameraImage, self.frame)
                    elif self.state == self.DroneState.focusingBody:
                        self.FocusBody(self.cameraImage, self.frame)
                    elif self.state == self.DroneState.approaching:
                        self.Approach(self.cameraImage, self.frame)


                else:
                    thread_tmp = threading.Thread(target=self._updateGUIImage,args=(image,))
                    thread_tmp.start()
                    time.sleep(0.03)                                                            
        except RuntimeError, e:
            print("[INFO] caught a RuntimeError")




    # Image - My Code
    def detection(self, faceImage, currentFrame):
        if currentFrame is not None:
            # Process Image
            datum = op.Datum()
            datum.cvInputData = currentFrame
            # Need to emplace and pop to get array for processing
            opWrapper.emplaceAndPop([datum])
            poseModel = op.PoseModel.BODY_25

            # 1L, 25L, 3L represents one person found in image. 2L, 25L, 3L would be 2 people and so on...
            if (self.recording == False and self.recordingFallen == False) and numpy.shape(datum.poseKeypoints) >= (
            1L, 25L, 3L):
                # Loop through persons detected and store keypoint information
                for person in range(len(datum.poseKeypoints)):
                    # Body point identification numbers obtained from
                    # Loops required as it was found that directly assigning body points was error prone
                    for n in op.getPosePartPairs(poseModel):
                        if n == 8:
                            self.target.setHip(n)
                            break
                    for n in op.getPosePartPairs(poseModel):
                        if n == 7:
                            self.target.setLWrist(n)
                            break
                    for n in op.getPosePartPairs(poseModel):
                        if n == 6:
                            self.target.setLElbow(n)
                            break
                    for n in op.getPosePartPairs(poseModel):
                        if n == 5:
                            self.target.setLShoulder(n)
                            break
                    for n in op.getPosePartPairs(poseModel):
                        if n == 4:
                            self.target.setRWrist(n)
                            break
                    for n in op.getPosePartPairs(poseModel):
                        if n == 3:
                            self.target.setRElbow(n)
                            break
                    for n in op.getPosePartPairs(poseModel):
                        if n == 2:
                            self.target.setRShoulder(n)
                            break
                    for n in op.getPosePartPairs(poseModel):
                        if n == 1:
                            self.target.setNeck(n)
                            break

                    # Set the coordinates for each of the required body points
                    # X and Y coordinates stored in separate variables for readability
                    self.target.setHipCoordinates(datum.poseKeypoints[person, self.target.getHip(), 0:2])
                    hipX = self.target.getHipCoordinates()[0]
                    hipY = self.target.getHipCoordinates()[1]


                    self.target.setLElbowCoordinates(datum.poseKeypoints[person, self.target.getLElbow(), 0:2])
                    lElbowX = self.target.getLElbowCoordinates()[0]
                    lElbowY = self.target.getLElbowCoordinates()[1]


                    self.target.setRElbowCoordinates(datum.poseKeypoints[person, self.target.getRElbow(), 0:2])
                    rElbowX = self.target.getRElbowCoordinates()[0]
                    rElbowY = self.target.getRElbowCoordinates()[1]


                    self.target.setNeckCoordinates(datum.poseKeypoints[person, self.target.getNeck(), 0:2])
                    neckX = self.target.getNeckCoordinates()[0]
                    neckY = self.target.getNeckCoordinates()[1]


                    # Search for waving people by checking if arms are raised above neck
                    # Also checks position of hip relative to neck to ensure that person is being viewed upright
                    if lElbowY > 0 and rElbowY > 0 and neckY > 0 and hipY > 0:
                        if ((lElbowY < neckY) and (rElbowY < neckY)) and (neckY < hipY):
                            # If a person is detected then send stop command and set distress id to person discovered
                            # Change state to focus on target and update GUI
                            if self.state == self.DroneState.searching:
                                print("Distress pose detected")
                                self.tello.send_command('stop')
                                self.target.setId(person)
                                self.waveDetected = True
                                self.state = self.DroneState.focusingBody
                                poseImage = Image.fromarray(datum.cvOutputData)
                                self._updateGUIImage(poseImage)
                                return 1
                                break
                    # Search for fallen people by checking the relative distance between the hip and neck
                    # Y-coordinates. If distance is small then person is likely in horizontal position (lying down)
                    # Same principle applies when looking at target from head-to-toe from low viewing angle
                    if ((abs(hipY - neckY) < 20) and ((hipY > 0) and (neckY > 0))):
                        if self.state == self.DroneState.searching:
                            print("Fallen detected")
                            self.tello.send_command('stop')
                            self.fallenDetected = True
                            self.state = self.DroneState.focusingBody
                            poseImage = Image.fromarray(datum.cvOutputData)
                            self._updateGUIImage(poseImage)
                            return 1
                            break

            # If recording is TRUE then the required keypoints must be set and stored in currentGesture array
            # For comparison with model gesture data later on
            if self.recording == True:
                self.target.setLWristCoordinates(datum.poseKeypoints[self.target.getId(), self.target.getLWrist(), 0:2])

                self.target.setLElbowCoordinates(datum.poseKeypoints[self.target.getId(), self.target.getLElbow(), 0:2])
                lElbowX = self.target.getLElbowCoordinates()[0]
                lElbowY = self.target.getLElbowCoordinates()[1]

                self.target.setLShoulderCoordinates(datum.poseKeypoints[self.target.getId(), self.target.getLShoulder(), 0:2])

                self.target.setRWristCoordinates(datum.poseKeypoints[self.target.getId(), self.target.getRWrist(), 0:2])

                self.target.setRElbowCoordinates(datum.poseKeypoints[self.target.getId(), self.target.getRElbow(), 0:2])
                rElbowX = self.target.getRElbowCoordinates()[0]
                rElbowY = self.target.getRElbowCoordinates()[1]

                self.target.setRShoulderCoordinates(datum.poseKeypoints[self.target.getId(), self.target.getRShoulder(), 0:2])

                # Each keypoint is checked to ensure that it is present in view
                # If a keypoint is not present it will be given [0,0] as coordinates which would skew comparison
                if self.target.getLWristCoordinates()[0] != 0 and self.target.getLElbowCoordinates()[0] != 0 and \
                        self.target.getLShoulderCoordinates()[0] != 0 and self.target.getRWristCoordinates()[0] != 0 and \
                        self.target.getRElbowCoordinates()[0] != 0 and self.target.getRShoulderCoordinates()[0] != 0:
                    self.currentGesture.append(self.target.getLWristCoordinates()) #lwristco
                    self.currentGesture.append(self.target.getLElbowCoordinates())#lelbow
                    self.currentGesture.append(self.target.getLShoulderCoordinates())#lshoul
                    self.currentGesture.append(self.target.getRWristCoordinates())#rwrist
                    self.currentGesture.append(self.target.getRElbowCoordinates())#relbo
                    self.currentGesture.append(self.target.getRShoulderCoordinates())#rshoul

                # if self.target.getLWristCoordinates()[0] != 0:
                #     self.currentGesture.append(self.target.getLWristCoordinates()) #lwristco
                # if self.target.getLElbowCoordinates()[0] != 0:
                #     self.currentGesture.append(self.target.getLElbowCoordinates())#lelbow
                # if self.target.getLShoulderCoordinates()[0] != 0:
                #     self.currentGesture.append(self.target.getLShoulderCoordinates())#lshoul
                # if self.target.getRWristCoordinates()[0] != 0:
                #     self.currentGesture.append(self.target.getRWristCoordinates())#rwrist
                # if self.target.getRElbowCoordinates()[0] != 0:
                #     self.currentGesture.append(self.target.getRElbowCoordinates())#relbo
                # if self.target.getRShoulderCoordinates()[0] != 0:
                #     self.currentGesture.append(self.target.getRShoulderCoordinates())#rshoul


            # If recordingFallen is TRUE then the required keypoints must be set and stored in currentFallenGesture
            # array until it reaches a predefined length (in frames). Recording will then store data into the
            # confirmFallenGesture array. The two will then be compared for similarity to check whether the fallen
            # person has moved.
            if self.recordingFallen == True:
                if len(self.currentFallenGesture) < 45:
                    self.currentFallenGesture.append(datum.poseKeypoints)
                elif len(self.confirmFallenGesture) < len(self.currentFallenGesture):
                    self.confirmFallenGesture.append(datum.poseKeypoints)
                else:
                    # Perform FastDTW on the two arrays and if they are a close match then
                    # the person is likely unconscious
                    fallenX = numpy.array(self.currentFallenGesture)
                    fallenX = numpy.vstack(fallenX).astype(numpy.float)
                    fallenX = fallenX.flatten()

                    fallenY = numpy.array(self.confirmFallenGesture)
                    fallenY = numpy.vstack(fallenY).astype(numpy.float)
                    fallenY = fallenY.flatten()

                    fallenDistance, fallenPath = fastdtw(fallenX, fallenY, dist=euclidean)

                    print(fallenDistance)

                    # Distance set to large amount for outdoor testing due to effects of wind on drone
                    # Indoor setting would permit < 80,000
                    if fallenDistance <= 250000:
                        print("Fallen person found")
                        self.state = self.DroneState.approaching
                    else:
                        self.state = self.DroneState.searching

                    self.recordingFallen = False
                    self.currentFallenGesture = []
                    self.confirmFallenGesture = []

            # Keep length of the current gesture no longer than the sample
            if len(self.currentGesture) > (len(self.distressGesture)):
                del self.currentGesture[0]

            for n in op.getPosePartPairs(poseModel):
                if n == 0:
                    self.target.setNose(n)
                    break

            # Check if a body can be seen
            if numpy.shape(datum.poseKeypoints) == (1L, 25L, 3L):
                self.target.setNoseCoordinates(datum.poseKeypoints[self.target.getId(), self.target.getNose(), 0:2])
                noseX = self.target.getNoseCoordinates()[0]
                noseY = self.target.getNoseCoordinates()[1]

                # Rectangle is at exact position of nose - radius is large so that it can be seen
                cv2.rectangle(currentFrame, (noseX, noseY), (noseX, noseY), (0, 0, 255), 25)
                datum.cvInputData = currentFrame



            # Convert into image to be displayed in GUI
            poseImage = Image.fromarray(datum.cvOutputData)


        # FastDTW
        # This works exceptionally well with shorter time series

        # If current array is >= length of recorded gesture then
        # if the array is bigger then discard the first part
        # then compare the two arrays

        if len(self.currentGesture) >= (len(self.distressGesture)):
            x = self.distressGesture
            x = x.flatten()

            y = numpy.array(self.currentGesture)
            y = numpy.vstack(y).astype(numpy.float)
            y = y.flatten()

            distance, path = fastdtw(x, y, dist=euclidean)

            print(distance)

            # Threshold chosen after testing - suitable for indoor and outdoor tests
            if distance <= 65000:
                print("PERSON IN DISTRESS FOUND")
                self.recording = False
                self.currentGesture = []
                self.state = self.DroneState.approaching

            if distance >= 100000:
                print("Person not in distress")
                self.recording = False
                self.currentGesture = []
                self.state = self.DroneState.searching


        img = numpy.array(poseImage)
        img = img[:, :, ::-1].copy()


        # Convert image to grayscale for feature detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        width = int(img.shape[1] * 60 / 100)
        height = int(img.shape[0] * 60 / 100)
        dimensions = (width, height)

        resizedGray = cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)

        if numpy.shape(datum.poseKeypoints) >= (1L, 25L, 3L):
            # Build lists of rectangles where features were detected within image
            self.faces = self.faceCascade.detectMultiScale(gray, 1.3, 4)

            # To enable body detection with Haar Cascades, uncomment the following line of code
            # bodies = self.bodyCascade.detectMultiScale(resizedGray, 1.01, 6)
            bodies = [] # Comment out if using Haar cascade for body detecion
        else:
            self.faces = []
            bodies = []


        # Define initial values for drone position and frame variables
        xPosFace = None
        yPosFace = None
        rectWidthFace = None
        rectHeightFace = None

        # Find faces
        for (x, y, w, h) in self.faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            xPosFace = x
            yPosFace = y
            rectWidthFace = w
            rectHeightFace = h


        # Find bodies
        for (x, y, w, h) in bodies:
            cv2.rectangle(img, (x, y), (x + w, y + h), (128, 0, 128), 2)
            xPosBody = x
            yPosBody = y
            rectWidthBody = w
            rectHeightBody = h

        # --------------------------------------------------------------------------------
        ##################################################################################
        # NOT CURRENTLY IMPLEMENTED AS OF THIS VERSION
        ##################################################################################
        # Canny edge detection attempt to be used to detection obstacles and avoid collisions

        # # Find Canny edges
        # edged = cv2.GaussianBlur(gray, (13, 13), 0)
        # edged = cv2.Canny(gray, 30, 200)
        #
        # contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #
        # if len(contours) > 15:
        #     cont = contours[0]
        #     #cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        #
        #     objX, objY, objW, objH = cv2.boundingRect(cont)
        #     cv2.rectangle(img, (objX, objY), (objX + objW, objY + objH), (0, 255, 0), 2)
        # # else:
        # #     self.tello.send_command('back 30')
        # #     time.sleep(2)
        # #     self.tello.send_command('cw 180')
        # #     time.sleep(1)
        # #     self.tello.send_command('forward 20')
        # #     time.sleep(1)

        ########################################################################################
        # ---------------------------------------------------------------------------------------------
        ########################################################################################

        # Invert colour channels from BGR to RGB
        #  for display in Tkinter
        b, g, r = cv2.split(img)
        img = cv2.merge((r, g, b))

        img = Image.fromarray(img)

        self._updateGUIImage(img)

        return 0


    #
    # Drone Behaviour
    #
    # While it does not have a body in view - search for one
    def Search(self, img, frame):
        # Expanding square formation
        if self.state == self.DroneState.searching:
            # While drone has not completed defined amount of search loops then
            # continue to search while looking for targets
            if self.searchLoops > 0:
                # Send command then initiates a wait for 4 seconds to allow drone
                # to complete given command
                self.tello.send_command('forward ' + str(self.searchDistance)) # Travel along edge of search square
                currentTime = time.time()
                while time.time() - currentTime < 4:
                    self.frame = self.tello.read() # Keep feed up to date
                    #If a target is detected then function should return as a state change will occur
                    if self.detection(self.cameraImage, self.frame) == 1:
                        return 0

                self.edgesTravelled += 1

                self.tello.rotate_cw(90) # Turn corner of search square
                currentTime = time.time()
                while time.time() - currentTime < 4:
                    self.frame = self.tello.read()
                    if self.detection(self.cameraImage, self.frame) == 1:
                        return 0

                # If drone has completed a loop then increase the distance it must travel
                # thus expanding the square
                if self.edgesTravelled >= 4:
                    self.searchDistance = self.searchDistance * 2
                    self.edgesTravelled = 0
                    self.searchLoops -= 1
            else:
                self.tello.send_command('land')


    # Focusing on body
    def FocusBody(self, img, frame):
        self.detection(self.cameraImage, self.frame)
        if self.target.neckCoordinates[0] > 0:
            # If the target's neck X-coordinate is below the center of the frame
            # minus a given threshold then the drone needs to turn CW to center the target in frame
            if (self.frameWidth / 2) <= self.target.neckCoordinates[0] - 200:
                self.focusX('CW')
                currentTime = time.time()
                while time.time() - currentTime < 3:
                    self.frame = self.tello.read()
                    self.detection(self.cameraImage, self.frame)

            elif (self.frameWidth / 2) >= self.target.neckCoordinates[0] + 200:
                self.focusX('CCW')
                currentTime = time.time()
                while time.time() - currentTime < 3:
                    self.frame = self.tello.read()
                    self.detection(self.cameraImage, self.frame)

            # If the target's neck Y-coordinate is below the center of the frame
            # minus a given threshold then the drone needs to ascend to center the target in frame
            elif (self.frameHeight / 2) <= self.target.neckCoordinates[1] - 200:
                self.focusY('DOWN')
                currentTime = time.time()
                while time.time() - currentTime < 3:
                    self.frame = self.tello.read()
                    self.detection(self.cameraImage, self.frame)

            elif (self.frameHeight / 2) >= self.target.neckCoordinates[1] + 200:
                self.focusY('UP')
                currentTime = time.time()
                while time.time() - currentTime < 3:
                    self.frame = self.tello.read()
                    self.detection(self.cameraImage, self.frame)

            # If all above are satisfied then target should be centered in frame and
            # drone can approach target
            else:
                if self.state != self.DroneState.approaching:
                    if self.waveDetected == True:
                        self.recording = True
                    elif self.fallenDetected == True:
                        self.recordingFallen = True


    # Approaching
    def Approach(self, img, frame):
        if self.state == self.DroneState.approaching:
            self.detection(self.cameraImage, self.frame)
            # If a face is detected then facial confirmation has been successful
            if len(self.faces) > 0:
                print("Face Confirmed")
                self.tello.send_command('land')
            # If a face has yet to be confirmed and target is still in view then
            # drone should approach target
            else:
                self.tello.send_command('forward 20')
                self.FocusBody(img, frame)
                currentTime = time.time()
                while time.time() - currentTime < 2:
                    self.frame = self.tello.read()
                    self.detection(self.cameraImage, self.frame)

                if self.target.neckCoordinates[0] <= 0:
                    telloHeight = self.tello.send_command('height?')
                    if telloHeight < 50:
                        self._sendingCommand('up 20')
                        while time.time() - currentTime < 2:
                            self.frame = self.tello.read()
                            self.detection(self.cameraImage, self.frame)
                    elif telloHeight > 80:
                        self._sendingCommand('down 20')
                        while time.time() - currentTime < 2:
                            self.frame = self.tello.read()
                            self.detection(self.cameraImage, self.frame)
                    self.state = self.DroneState.searching

        # Idle
        elif self.state == self.DroneState.idle:
            self.tello.send_command('speed 0')


    def focusX(self, direction):
        if direction == 'CW':
            self.tello.rotate_cw(self.degree / 2)
        elif direction == 'CCW':
            self.tello.rotate_ccw(self.degree / 2)

    def focusY(self, direction):
        if direction == 'UP':
            self.tello.send_command('up 20')
        elif direction == 'DOWN':
            self.tello.send_command('down 20')



    def _updateGUIImage(self,image):
        """
        Main operation to initial the object of image,and update the GUI panel 
        """  
        image = ImageTk.PhotoImage(image)
        # if the panel none ,we need to initial it
        if self.panel is None:
            self.panel = tki.Label(image=image)
            self.panel.image = image
            self.panel.pack(side="left", padx=10, pady=10)
        # otherwise, simply update the panel
        else:
            self.panel.configure(image=image)
            self.panel.image = image

    def _sendingCommand(self):
        """
        start a while loop that sends 'command' to tello every 5 second
        """    

        while True:
            self.tello.send_command('command')        
            time.sleep(5)

    def _setQuitWaitingFlag(self):  
        """
        set the variable as TRUE,it will stop computer waiting for response from tello  
        """       
        self.quit_waiting_flag = True        
   
    def openCmdWindow(self):
        """
        open the cmd window and initial all the button and text
        """        
        panel = Toplevel(self.root)
        panel.wm_title("Command Panel")

        # create text input entry
        text0 = tki.Label(panel,
                          text='This Controller map keyboard inputs to Tello control commands\n'
                               'Adjust the trackbar to reset distance and degree parameter',
                          font='Helvetica 10 bold'
                          )
        text0.pack(side='top')

        text1 = tki.Label(panel, text=
                          'W - Move Tello Up\t\t\tArrow Up - Move Tello Forward\n'
                          'S - Move Tello Down\t\t\tArrow Down - Move Tello Backward\n'
                          'A - Rotate Tello Counter-Clockwise\tArrow Left - Move Tello Left\n'
                          'D - Rotate Tello Clockwise\t\tArrow Right - Move Tello Right',
                          justify="left")
        text1.pack(side="top")

        self.btn_landing = tki.Button(
            panel, text="Land", relief="raised", command=self.telloLanding)
        self.btn_landing.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        self.btn_takeoff = tki.Button(
            panel, text="Takeoff", relief="raised", command=self.telloTakeOff)
        self.btn_takeoff.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        # binding arrow keys to drone control
        self.tmp_f = tki.Frame(panel, width=100, height=2)
        self.tmp_f.bind('<KeyPress-w>', self.on_keypress_w)
        self.tmp_f.bind('<KeyPress-s>', self.on_keypress_s)
        self.tmp_f.bind('<KeyPress-a>', self.on_keypress_a)
        self.tmp_f.bind('<KeyPress-d>', self.on_keypress_d)
        self.tmp_f.bind('<KeyPress-Up>', self.on_keypress_up)
        self.tmp_f.bind('<KeyPress-Down>', self.on_keypress_down)
        self.tmp_f.bind('<KeyPress-Left>', self.on_keypress_left)
        self.tmp_f.bind('<KeyPress-Right>', self.on_keypress_right)
        self.tmp_f.pack(side="bottom")
        self.tmp_f.focus_set()

        self.btn_landing = tki.Button(
            panel, text="Flip", relief="raised", command=self.openFlipWindow)
        self.btn_landing.pack(side="bottom", fill="both",
                              expand="yes", padx=10, pady=5)

        self.distance_bar = Scale(panel, from_=0.02, to=5, tickinterval=0.01, digits=3, label='Distance(m)',
                                  resolution=0.01)
        self.distance_bar.set(0.2)
        self.distance_bar.pack(side="left")

        self.btn_distance = tki.Button(panel, text="Reset Distance", relief="raised",
                                       command=self.updateDistancebar,
                                       )
        self.btn_distance.pack(side="left", fill="both",
                               expand="yes", padx=10, pady=5)

        self.degree_bar = Scale(panel, from_=1, to=360, tickinterval=10, label='Degree')
        self.degree_bar.set(30)
        self.degree_bar.pack(side="right")

        self.btn_distance = tki.Button(panel, text="Reset Degree", relief="raised", command=self.updateDegreebar)
        self.btn_distance.pack(side="right", fill="both",
                               expand="yes", padx=10, pady=5)

    def openFlipWindow(self):
        """
        open the flip window and initial all the button and text
        """
        
        panel = Toplevel(self.root)
        panel.wm_title("Gesture Recognition")

        self.btn_flipl = tki.Button(
            panel, text="Flip Left", relief="raised", command=self.telloFlip_l)
        self.btn_flipl.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_flipr = tki.Button(
            panel, text="Flip Right", relief="raised", command=self.telloFlip_r)
        self.btn_flipr.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_flipf = tki.Button(
            panel, text="Flip Forward", relief="raised", command=self.telloFlip_f)
        self.btn_flipf.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)

        self.btn_flipb = tki.Button(
            panel, text="Flip Backward", relief="raised", command=self.telloFlip_b)
        self.btn_flipb.pack(side="bottom", fill="both",
                            expand="yes", padx=10, pady=5)
       
    def takeSnapshot(self):
        """
        save the current frame of the video as a jpg file and put it into outputpath
        """

        # grab the current timestamp and use it to construct the filename
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))

        p = os.path.sep.join((self.outputPath, filename))

        # save the file
        cv2.imwrite(p, cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
        print("[INFO] saved {}".format(filename))


    def pauseVideo(self):
        """
        Toggle the freeze/unfreze of video
        """
        if self.btn_pause.config('relief')[-1] == 'sunken':
            self.btn_pause.config(relief="raised")
            self.tello.video_freeze(False)
        else:
            self.btn_pause.config(relief="sunken")
            self.tello.video_freeze(True)

    def telloTakeOff(self):
        return self.tello.takeoff()                

    def telloLanding(self):
        return self.tello.land()

    def telloFlip_l(self):
        return self.tello.flip('l')

    def telloFlip_r(self):
        return self.tello.flip('r')

    def telloFlip_f(self):
        return self.tello.flip('f')

    def telloFlip_b(self):
        return self.tello.flip('b')

    def telloCW(self, degree):
        return self.tello.rotate_cw(degree)

    def telloCCW(self, degree):
        return self.tello.rotate_ccw(degree)

    def telloMoveForward(self, distance):
        return self.tello.move_forward(distance)

    def telloMoveBackward(self, distance):
        return self.tello.move_backward(distance)

    def telloMoveLeft(self, distance):
        return self.tello.move_left(distance)

    def telloMoveRight(self, distance):
        return self.tello.move_right(distance)

    def telloUp(self, dist):
        return self.tello.move_up(dist)

    def telloDown(self, dist):
        return self.tello.move_down(dist)

    def updateTrackBar(self):
        self.my_tello_hand.setThr(self.hand_thr_bar.get())

    def updateDistancebar(self):
        self.distance = self.distance_bar.get()
        print 'reset distance to %.1f' % self.distance

    def updateDegreebar(self):
        self.degree = self.degree_bar.get()
        print 'reset distance to %d' % self.degree

    def on_keypress_w(self, event):
        print "up %d m" % self.distance
        self.telloUp(self.distance)

    def on_keypress_s(self, event):
        print "down %d m" % self.distance
        self.telloDown(self.distance)

    def on_keypress_a(self, event):
        print "ccw %d degree" % self.degree
        self.tello.rotate_ccw(self.degree)

    def on_keypress_d(self, event):
        print "cw %d m" % self.degree
        self.tello.rotate_cw(self.degree)

    def on_keypress_up(self, event):
        print "forward %d m" % self.distance
        self.telloMoveForward(self.distance)

    def on_keypress_down(self, event):
        print "backward %d m" % self.distance
        self.telloMoveBackward(self.distance)

    def on_keypress_left(self, event):
        print "left %d m" % self.distance
        self.telloMoveLeft(self.distance)

    def on_keypress_right(self, event):
        print "right %d m" % self.distance
        self.telloMoveRight(self.distance)

    def on_keypress_enter(self, event):
        if self.frame is not None:
            self.registerFace()
        self.tmp_f.focus_set()

    def onClose(self):
        """
        set the stop event, cleanup the camera, and allow the rest of
        
        the quit process to continue
        """
        # TEMPORARY WEBCAM CODE ------------------------------------------
        self.webcam.release()

        print("[INFO] closing...")
        self.stopEvent.set()
        del self.tello
        self.root.quit()

