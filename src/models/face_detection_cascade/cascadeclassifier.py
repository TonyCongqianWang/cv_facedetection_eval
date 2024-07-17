import numpy as np
import cv2 as cv

class CascadeClassifier:
    def __init__(self, modelPath, inputSize=[320, 320]):
        self._model = cv.CascadeClassifier(modelPath)


    @property
    def name(self):
        return self.__class__.__name__

    def setInputSize(self, input_size):
        pass

    def infer(self, image):
        faces = self._model.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)
        # Todo convert output format
        return np.array([]) if faces[1] is None else faces[1]
