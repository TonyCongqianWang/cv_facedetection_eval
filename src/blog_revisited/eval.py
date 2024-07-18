import argparse
from ast import For
from genericpath import isdir, isfile
import cv2
import sys
import os
import time
from pathlib import Path

models_dir = os.path.join(os.path.dirname(__file__), "../models")

parser = argparse.ArgumentParser("Simple inference with time evaluation on single image")
parser.add_argument('image_path', type=str,
                    help='Usage: Set input to a certain image or image directory.')
parser.add_argument('model', type=str,
                    help='Usage: Set model type and config.')
parser.add_argument('--out_dir', type=str,
                    help='Usage: Set out-dir of result imges.', default="./results/")
parser.add_argument('--num_threads', type=int, help="Usage: Set number of threads (1-4)", choices=[1,2,3,4], default=1)
args = parser.parse_args()

cv2.setNumThreads(args.num_threads)
os.makedirs(args.out_dir, exist_ok=True)

## To ensure same parameters are used for evaluation on WiderFace dataface both scripts the model wrapper objects created with default parameters

def get_cascade(model_path):
    cascade_dir = os.path.join(models_dir, "face_detection_cascade")
    sys.path.append(cascade_dir)
    from cascadeclassifier import CascadeClassifier
    model = CascadeClassifier(model_path, is_test=False, scaleFactor=1.1, minNeighbors=2)
    return model

def get_yunet(model_path):
    yunet_dir = os.path.join(models_dir, "face_detection_yunet")
    sys.path.append(yunet_dir)
    from yunet import YuNet
    model = YuNet(modelPath=model_path,
              confThreshold=0.9,
              nmsThreshold=0.3,
              backendId=3, # OpenCv Backend
              targetId=0) # CPU Device
    return model

def detect_on_img(img, iters, model, convert_to_gray):
    h, w, _ = img.shape
    model.setInputSize([w, h])
    if convert_to_gray:
        # Convert into grayscale
        det_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Starting time counter after conversion -> assumption is we already get grayscale images from device
    else:
        det_img = img.copy()
   
    # Warmup
    for _ in range(iters[0]):
        results = model.infer(det_img)
    # Inference
    tic = time.perf_counter()    
    for _ in range(iters[1]):
        results = model.infer(det_img)
    toc = time.perf_counter()
    return results, (toc - tic)/iters[1]

def save_result(img, results, img_name, img_time):
    out_file = os.path.join(args.out_dir, f"{img_name}_threads_{args.num_threads}_time.txt")
    with open(out_file, "w") as f: 
        img_time *= 1000
        print(f"Elapsed ms / img: {img_time:.1f}")
        print(f"Elapsed ms / img: {img_time:.1f}", file=f)
    out_file = os.path.join(args.out_dir,  f"{img_name}_detection.jpg")
    for face in results:
        x, y, w, h = [int(v) for v in face[:4]]
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 1)

    cv2.imwrite(out_file, img)
    #print(results)

convert_to_gray = False

if "cascade" in args.model:
    model = get_cascade(args.model)
    convert_to_gray = True
elif "yunet" in args.model:
    model = get_yunet(args.model)
else:
    raise ValueError("Unknown Model")

if os.path.isfile(args.image_path):
    img_paths = [args.image_path]
elif os.path.isdir(args.image_path):
    directory = Path(args.image_path)
    img_paths = directory.glob("*.jpg")
else:
    raise ValueError("Path does not exist")

for img_path in img_paths:
    img_name = Path(img_path).stem
    print(img_name)

    img = cv2.imread(str(img_path))
    iters = (30, 300)

    results, img_time = detect_on_img(img, iters, model, convert_to_gray)
    save_result(img, results, img_name, img_time)
