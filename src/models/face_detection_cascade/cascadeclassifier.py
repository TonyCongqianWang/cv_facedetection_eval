import numpy as np
import cv2 as cv

class CascadeClassifier:
    def __init__(self, modelPath, outputRejectLevels = True, scaleFactor=1.1, minNeighbors=3):
        self._scalefactor = scaleFactor
        self._minNeighbours = minNeighbors
        self._model = cv.CascadeClassifier(modelPath)
        self._outputRejectLevels = outputRejectLevels

    @property
    def name(self):
        return self.__class__.__name__

    def setInputSize(self, input_size):
        pass

    def infer(self, image):
        if self._outputRejectLevels:
            faces, _, levelWeights  =  self._model.detectMultiScale3(image, scaleFactor=self._scalefactor, minNeighbors=self._minNeighbours, outputRejectLevels=True)
            results = []
            for face, confidence in zip(faces, levelWeights):
                results.append(list(face) + [0 for _ in range(10)] + [confidence])
        else:
            faces = self._model.detectMultiScale(image, scaleFactor=self._scalefactor, minNeighbors=self._minNeighbours)
            results = [list(face) + [0 for _ in range(10)] + [0.5] for face in faces]
        
        return np.array(results)
