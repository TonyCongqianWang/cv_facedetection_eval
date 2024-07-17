import argparse
import cv2
import time

parser = argparse.ArgumentParser("Simple inference with time evaluation on single image")
parser.add_argument('image-path', type=str,
                    help='Usage: Set input to a certain image.')
parser.add_argument('model', type=str,
                    help='Usage: Set model type and config.')
parser.add_argument('--out-dir', type=str,
                    help='Usage: Set out-dir of result imges.', default="./result_imgs")
args = parser.parse_args()

def detect_cascade(img, k, model_path):
    face_cascade = cv2.CascadeClassifier(model_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Starting time counter after conversion -> assumption is we already get grayscale images from device
    tic = time.perf_counter()
    # Detect faces
    for i in range(1, k):
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    toc = time.perf_counter()
    print("Elapsed:", toc - tic)
    export_result_img(img, faces)


def detect_yunet(img, k, model_path):
    from yunet import YuNet
    model = YuNet(modelPath=model_path,
              inputSize=[320, 320],
              confThreshold=0.9,
              nmsThreshold=0.3,
              topK=5000,
              backendId=3, # OpenCv Backend
              targetId=0) # CPU Device
    h, w, _ = img.shape
    # Inference
    model.setInputSize([w, h])
    tic = time.perf_counter()
    for i in range(1, k):
        results = model.infer(img)
    toc = time.perf_counter()
    print("Elapsed:", toc - tic)
    export_result_img(img, results)

def export_result_img(img, faces):
    # TODO
    pass


img = cv2.imread(args.image_path)
k = 100

if "cascade" in args.model:
    detect_cascade(img, k, args.model)
elif "yunet" in args.model:
    detect_yunet(img, k, args.model)
else:
    raise ValueError("Unknown Model")