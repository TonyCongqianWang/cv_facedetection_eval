# OpenCV Face Detection: Cascade Classifier vs. YuNet Revisited

Comparison of the two popular OpenCV implementations for fast face detection on CPU. Inspired by [OpenCv Blog Entry](https://opencv.org/blog/opencv-face-detection-cascade-classifier-vs-yunet/).

## Extending Original Tests

Extending tests by using multiple configurations for both detectors and measuring inference time on two different machines (see below). For YuNet uint8 and fp32 weights are used while for the CascadeClassifier four haar-feature based and two lbp-feature based configurations are used. All configurations are taken from the opencv repositories. Images are resized crops from the WiderFace dataset, similar to the ones used in the [original blog post](https://opencv.org/blog/opencv-face-detection-cascade-classifier-vs-yunet/).

Laptop Info:
Architecture: x86-64 
OS: Windows 11 Pro Version 10.0.22631 Build 22631
Processor: 12th Gen Intel(R) Core(TM) i7-1255U, 1700 MHz, 10(12 logical) Cores

Raspberry Pi Info:
Architecture: aarch64
OS: Linux raspberrypi 6.1.0-rpi8-rpi-v8 (Debian Bookworm)
Processor: Cortex-A72, 1500 MHz, 4 Cores

### Results

While YuNet generally seems to be more accurate at detecting faces, when limiting the faces to upright frontal faces, the cascades seem to do a reasonable job that is not far off. While YuNet performs much faster on my Laptop, the results on the raspberry pi show that the lbp cascades are faster when using 4 threads. However, the quantized version of YuNet performed slower than the fp32 version. This indicates, that there is currently no optimization for quantized neural networks compatible with the aarch64 architecture. With the use of NEON simd instructions, the unit8 processing time should be lower by about a factor of 4. Surprisingly, the times measures on my raspberry pi were lower than the ones reported on the official [YuNet github](https://github.com/ShiqiYu/libfacedetection). This, combined with the fact that YuNet seems to not improve much with the use of 4 threads, indicates that YuNet in opencv-python is actually cheating and uses more threads than allowed!

See time_comparison_pi.md and time_comparison_laptop.md for result tables.

## Evaluation on WiderFace

Using the [opencv model zoo eval tool](https://github.com/opencv/opencv_zoo/blob/main/tools/eval) to calculate the scores of each detector on widerface dataset. Note: YuNet model was trained on the WiderFace train-set which is annotated consistently with the val-set and contains similar face types. Cascades were trained on much smaller datasets with potentially different annotation guidelines and focus specifically on frontal upright faces. Additionally cascade type models are bad at outputting a smooth range of confidence scores. Both factors will result in lower scores for the cascade models. To run the eval tool, you must install the requirements with pip install -r requirements_wider_eval.txt and download the directories WIDER_val, wider_face_split and eval_tools from WiderFace and put them in data/WiderFaceset

### Results

AP score on WIDER FACE val-set - no resizing.

For cascade_lbp_improved the boxes were padded.

|Model | Easy Set | Medium Set | Hard Set |
|-------------|--------|----------|--------|
|yunet  | 0.884 | 0.866 | 0.750 |
|yunet_q  | 0.881 | 0.862 | 0.750 |
|cascade_haar  | 0.215 | 0.260 | 0.144 |
|cascade_haar_alt  | 0.373 | 0.382 | 0.211 |
|cascade_haar_alt2  | 0.371 | 0.380 | 0.210 |
|cascade_haar_alt_tree  | 0.367 | 0.378 | 0.209 |
|cascade_lbp  | 0.202 | 0.227 | 0.118 |
|cascade_lbp_improved  | 0.282 | 0.154 | 0.064 |

## Conclusion

Depending on which configuration you use, the cascades an have very different performances, there is almost no reason to use haarcascades since they are very slow and less accurate than YuNet. LbpCascades however can be slightly faster. If you are running opencv on a x86 machine with many cores or a gpu, there YuNet is always the better choice than using the outdated cascades. However on a raspberry pi cascades can be slightly faster than YuNet. YuNet is far more accurate in general, but depending on your use-case cascades can have sufficient accuracy. The "improved" lbp cascade does seem to be a slight improvement for upright frontal faces as it has slightly lower runtime and better scores on upright faces. But the overall performance seems to be lower.
