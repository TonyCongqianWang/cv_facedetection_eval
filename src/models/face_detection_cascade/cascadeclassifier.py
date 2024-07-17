import numpy as np
import cv2 as cv
import math

class CascadeClassifier:
    def __init__(self, modelPath, is_test=True, input_size=None, initial_scale=1, scaleFactor=1.1, minNeighbors=3):
        self._scalefactor = scaleFactor
        self._minNeighbours = minNeighbors
        self._model = cv.CascadeClassifier(modelPath)
        self._is_test = is_test
        self._initial_scale = initial_scale
        _ = input_size

    @property
    def name(self):
        return self.__class__.__name__

    def setInputSize(self, input_size):
        _ = input_size

    def infer(self, image):
        if self._is_test:
            if len(image.shape) > 2 and image.shape[2] > 1:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            if self._initial_scale != 1:
                factor = self._initial_scale
                image = cv.resize(image, dsize=None, fx=factor, fy=factor)
            faces, _, levelWeights  =  self._model.detectMultiScale3(image, scaleFactor=self._scalefactor, minNeighbors=1, outputRejectLevels=True)
            results = []
            for face, confidence in zip(faces, levelWeights):
                #confidence = 1/(1 + math.exp(-confidence/2))
                results.append(list(face) + [0 for _ in range(10)] + [confidence])
        else:
            faces = self._model.detectMultiScale(image, scaleFactor=self._scalefactor, minNeighbors=self._minNeighbours)
            results = [list(face) + [0 for _ in range(10)] + [0.5] for face in faces]
        
        return np.empty(shape=(0, 5)) if len(results) == 0 else np.array(results)