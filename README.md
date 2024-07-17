# OpenCV Face Detection: Cascade Classifier vs. YuNet Revisited

Comparison of the two popular OpenCV implementations for fast face detection on CPU. Inspired by [OpenCv Blog Entry](https://opencv.org/blog/opencv-face-detection-cascade-classifier-vs-yunet/).

## Extending Original Tests

Extending tests by using multiple configurations for both detectors and measuring inference time on two different machines (see below). For YuNet uint8 and fp32 weights are used while for the CascadeClassifier one haar-feature based and two lbp-feature based configurations are used. All configurations are taken from the opencv repositories. Images are resized crops from the WiderFace dataset, similar to the ones used in the [original blog post](https://opencv.org/blog/opencv-face-detection-cascade-classifier-vs-yunet/).

Laptop Info:
Architecture: x86-64 
OS: Windows 11 Pro Version 10.0.22631 Build 22631
Processor: 12th Gen Intel(R) Core(TM) i7-1255U, 1700 MHz, 10(12 logical) Cores

Raspberry Pi Info:
Architecture: aarch64
OS: Linux raspberrypi 6.1.0-rpi8-rpi-v8 (Debian Bookworm)
Processor: Cortex-A72, 1500 MHz, 4 Cores


## Evaluation on WiderFace

Using the a [python implementation](https://github.com/wondervictor/WiderFace-Evaluation) of the WiderFace evaluation tool to calculate the scores of each detector.