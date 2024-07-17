import argparse
import cv2
import sys
import os
import time

models_dir = os.path.join("..", "..") #TODO

parser = argparse.ArgumentParser("Simple inference with time evaluation on single image")
parser.add_argument('image-path', type=str,
                    help='Usage: Set input to a certain image.')
parser.add_argument('model', type=str,
                    help='Usage: Set model type and config.')
parser.add_argument('--out-dir', type=str,
                    help='Usage: Set out-dir of result imges.', default="./result_imgs")
parser.add_argument('-num-threads', type=int, help="Usage: Set number of threads (1-4)", choices=[1,2,3,4])
args = parser.parse_args()

cv2.setNumThreads(args.num_threads)

## To ensure same parameters are used for evaluation on WiderFace dataface both scripts the model wrapper objects created with default parameters

def detect_cascade(img, k, model_path):
    cascade_dir = os.path.join(models_dir, "face_detection_cascade")
    sys.path.append(cascade_dir)
    from cascadeclassifier import CascadeClassifier
    model = CascadeClassifier(model_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Starting time counter after conversion -> assumption is we already get grayscale images from device
    tic = time.perf_counter()
    # Inference
    for i in range(1, k):
        results = model.infer(gray)
    toc = time.perf_counter()
   
    save_result(img, results, toc - tic)


def detect_yunet(img, k, model_path):
    yunet_dir = os.path.join(models_dir, "face_detection_yunet")
    sys.path.append(yunet_dir)
    from yunet import YuNet
    model = YuNet(modelPath=model_path,
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
    save_result(img, results, toc - tic)

def save_result(img, results, elapsed):
    out_file = os.path.join( args.out_dir, "/elapsed_time.txt")
    with open(out_file) as f: 
        print("Elapsed:", elapsed)
        print("Elapsed:",elapsed, file = f)
    out_file = os.path.join( args.out_dir, "/detection_img.jpg")
    save_img = img.copy() #TODO
    cv2.imwrite(save_img, img)


img = cv2.imread(args.image_path)
k = 100

if "cascade" in args.model:
    detect_cascade(img, k, args.model)
elif "yunet" in args.model:
    detect_yunet(img, k, args.model)
else:
    raise ValueError("Unknown Model")