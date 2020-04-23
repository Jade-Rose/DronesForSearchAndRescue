class TargetInDistess():
    def __init__(self):
        self.idOfPersonInDistress = 0

        self.hipCoordinates = [0,0]
        self.lWristCoordinates = [0,0]
        self.lElbowCoordinates = [0,0]
        self.lShoulderCoordinates = [0,0]
        self.rWristCoordinates = [0,0]
        self.rElbowCoordinates = [0,0]
        self.rShoulderCoordinates = [0,0]
        self.neckCoordinates = [0,0]
        self.noseCoordinates = [0,0]

        self.hip = 0    # 8
        self.lWrist = 0  # 7
        self.lElbow = 0  # 6
        self.lShoulder = 0  # 5
        self.rWrist = 0  # 4
        self.rElbow = 0  # 3
        self.rShoulder = 0  # 2
        self.neck = 0   # 1
        self.nose = 0   # 0

    def getId(self):
        return self.idOfPersonInDistress

    def getHipCoordinates(self):
        return self.hipCoordinates

    def getLWristCoordinates(self):
        return self.lWristCoordinates

    def getLElbowCoordinates(self):
        return self.lElbowCoordinates

    def getLShoulderCoordinates(self):
        return self.lShoulderCoordinates

    def getRWristCoordinates(self):
        return self.rWristCoordinates

    def getRElbowCoordinates(self):
        return self.rElbowCoordinates

    def getRShoulderCoordinates(self):
        return self.rShoulderCoordinates

    def getNeckCoordinates(self):
        return self.neckCoordinates

    def getNoseCoordinates(self):
        return self.noseCoordinates

    def getHip(self):
        return self.hip

    def getLWrist(self):
        return self.lWrist

    def getLElbow(self):
        return self.lElbow

    def getLShoulder(self):
        return self.lShoulder

    def getRWrist(self):
        return self.rWrist

    def getRElbow(self):
        return self.rElbow

    def getRShoulder(self):
        return self.rShoulder

    def getNeck(self):
        return self.neck

    def getNose(self):
        return self.nose

    def setId(self, newId):
        self.idOfPersonInDistress = newId

    def setHipCoordinates(self, newCoordinates):
        self.hipCoordinates = newCoordinates

    def setLWristCoordinates(self, newCoordinates):
        self.lWristCoordinates = newCoordinates

    def setLElbowCoordinates(self, newCoordinates):
        self.lElbowCoordinates = newCoordinates

    def setLShoulderCoordinates(self, newCoordinates):
        self.lShoulderCoordinates = newCoordinates

    def setRWristCoordinates(self, newCoordinates):
        self.rWristCoordinates = newCoordinates

    def setRElbowCoordinates(self, newCoordinates):
        self.rElbowCoordinates = newCoordinates

    def setRShoulderCoordinates(self, newCoordinates):
        self.rShoulderCoordinates = newCoordinates

    def setNeckCoordinates(self, newCoordinates):
        self.neckCoordinates = newCoordinates

    def setNoseCoordinates(self, newCoordinates):
        self.noseCoordinates = newCoordinates

    def setHip(self, newIndex):
        self.hip = newIndex

    def setLWrist(self, newIndex):
        self.lWrist = newIndex

    def setLElbow(self, newIndex):
        self.lElbow = newIndex

    def setLShoulder(self, newIndex):
        self.lShoulder = newIndex

    def setRWrist(self, newIndex):
        self.rWrist = newIndex

    def setRElbow(self, newIndex):
        self.rElbow = newIndex

    def setRShoulder(self, newIndex):
        self.rShoulder = newIndex

    def setNeck(self, newIndex):
        self.neck = newIndex

    def setNose(self, newIndex):
        self.nose = newIndex