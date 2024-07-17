import os
import sys
import argparse

import numpy as np
import cv2 as cv

from datasets import DATASETS

root_dir = os.path.join("..", "..")
sys.path.append(root_dir)
from models import MODELS

parser = argparse.ArgumentParser("Evaluation with OpenCV on different models in the zoo.")
parser.add_argument("--model", "-m", type=str, required=True, help="model name")
parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name")
parser.add_argument("--dataset_root", "-dr", type=str, required=True, help="Root directory of given dataset")
args = parser.parse_args()

models = dict(
    yunet=dict(
        name="YuNet",
        topic="face_detection",
        modelPath=os.path.join(root_dir, "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
        topK=5000,
        confThreshold=0.3,
        nmsThreshold=0.45),
    yunet_q=dict(
        name="YuNet",
        topic="face_detection",
        modelPath=os.path.join(root_dir, "models/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx"),
        topK=5000,
        confThreshold=0.3,
        nmsThreshold=0.45),
    cascade_haar=dict(
        name="HaarCascade",
        topic="face_detection"),    
    cascade_haar_alt=dict(
        name="HaarCascade",
        topic="face_detection"),
    cascade_lbp=dict(
        name="LbpCascade",
        topic="face_detection"),
    cascade_lbp_improved=dict(
        name="LbpCascade",
        topic="face_detection"),
)

datasets = dict(
        widerface=dict(
            name="WIDERFace",
            topic="face_detection"),
)

def main(args):
    # Instantiate model
    model_key = args.model.lower()
    assert model_key in models

    model_name = models[model_key].pop("name")
    model_topic = models[model_key].pop("topic")
    model_handler, _ = MODELS.get(model_name)
    model = model_handler(**models[model_key])

    # Instantiate dataset
    dataset_key = args.dataset.lower()
    assert dataset_key in datasets

    dataset_name = datasets[dataset_key].pop("name")
    dataset_topic = datasets[dataset_key].pop("topic")
    dataset = DATASETS.get(dataset_name)(root=args.dataset_root, **datasets[dataset_key])

    # Check if model_topic matches dataset_topic
    assert model_topic == dataset_topic

    # Run evaluation
    dataset.eval(model)
    dataset.print_result()

if __name__ == "__main__":
    main(args)
