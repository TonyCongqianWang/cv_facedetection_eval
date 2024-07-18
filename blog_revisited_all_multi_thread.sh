python src/blog_revisited/eval.py data/test_imgs/ src/models/face_detection_yunet/face_detection_yunet_2023mar.onnx --out_dir results/blog_revisit_results/yunet --num_threads 4
python src/blog_revisited/eval.py data/test_imgs/ src/models/face_detection_yunet/face_detection_yunet_2023mar_int8.onnx --out_dir results/blog_revisit_results/yunet_q --num_threads 4

python src/blog_revisited/eval.py data/test_imgs/ src/models/face_detection_cascade/lbpcascade_frontalface.xml --out_dir results/blog_revisit_results/cascade_lbp --num_threads 4
python src/blog_revisited/eval.py data/test_imgs/ src/models/face_detection_cascade/lbpcascade_frontalface_improved.xml --out_dir results/blog_revisit_results/cascade_lbp_improved --num_threads 4

python src/blog_revisited/eval.py data/test_imgs/ src/models/face_detection_cascade/haarcascade_frontalface_default.xml --out_dir results/blog_revisit_results/cascade_haar --num_threads 4
python src/blog_revisited/eval.py data/test_imgs/ src/models/face_detection_cascade/haarcascade_frontalface_alt.xml --out_dir results/blog_revisit_results/cascade_haar_alt --num_threads 4
python src/blog_revisited/eval.py data/test_imgs/ src/models/face_detection_cascade/haarcascade_frontalface_alt2.xml --out_dir results/blog_revisit_results/cascade_haar_alt2 --num_threads 4
python src/blog_revisited/eval.py data/test_imgs/ src/models/face_detection_cascade/haarcascade_frontalface_alt_tree.xml --out_dir results/blog_revisit_results/cascade_haar_alt_tree --num_threads 4