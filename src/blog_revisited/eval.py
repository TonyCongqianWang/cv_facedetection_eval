import argparse
import cv2
import sys
import os
import time

models_dir = os.path.join(os.path.dirname(__file__), "../models")

parser = argparse.ArgumentParser("Simple inference with time evaluation on single image")
parser.add_argument('image_path', type=str,
                    help='Usage: Set input to a certain image.')
parser.add_argument('model', type=str,
                    help='Usage: Set model type and config.')
parser.add_argument('--out_dir', type=str,
                    help='Usage: Set out-dir of result imges.', default="./results/")
parser.add_argument('-num_threads', type=int, help="Usage: Set number of threads (1-4)", choices=[1,2,3,4])
args = parser.parse_args()

cv2.setNumThreads(args.num_threads)
os.makedirs(args.out_dir, exist_ok=True)

## To ensure same parameters are used for evaluation on WiderFace dataface both scripts the model wrapper objects created with default parameters

def get_cascade(model_path):
    cascade_dir = os.path.join(models_dir, "face_detection_cascade")
    sys.path.append(cascade_dir)
    from cascadeclassifier import CascadeClassifier
    model = CascadeClassifier(model_path, outputRejectLevels=True)
    return model

def get_yunet(model_path):
    yunet_dir = os.path.join(models_dir, "face_detection_yunet")
    sys.path.append(yunet_dir)
    from yunet import YuNet
    model = YuNet(modelPath=model_path,
              backendId=3, # OpenCv Backend
              targetId=0) # CPU Device
    h, w, _ = img.shape
    # Inference
    model.setInputSize([w, h])
    return model

def detect_on_img(img, k, model, convert_to_gray):
    if convert_to_gray:
        # Convert into grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Starting time counter after conversion -> assumption is we already get grayscale images from device
    tic = time.perf_counter()
    # Inference
    for _ in range(k):
        results = model.infer(img)
    toc = time.perf_counter()
   
    save_result(img, results, toc - tic)

def save_result(img, results, elapsed):
    out_file = os.path.join(args.out_dir, "elapsed_time.txt")
    with open(out_file, "w") as f: 
        print("Elapsed s / img:", elapsed/k)
        print("Elapsed s / img:", elapsed/k, file = f)
    out_file = os.path.join(args.out_dir, "detection_img.jpg")
    save_img = img.copy() #TODO
    cv2.imwrite(out_file, save_img)
    print([list(r[:4]) + [r[14]] for r in results])


img = cv2.imread(args.image_path)
k = 1
convert_to_gray = False

if "cascade" in args.model:
    model = get_cascade(args.model)
    convert_to_gray = True
elif "yunet" in args.model:
    model = get_yunet(args.model)
else:
    raise ValueError("Unknown Model")

detect_on_img(img, k, model, convert_to_gray)