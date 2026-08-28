# Model checkpoints

Model checkpoints are intentionally excluded from version control.

The original archive contained two YOLOv5 checkpoints, `best.pt` and `last.pt`, approximately 14.3 MB each. They were not published because generated model binaries are reproducible outputs and can have separate redistribution and security considerations.

To recreate the detector checkpoint, run `notebooks/04_yolov5_shap.ipynb`. By default, the notebook writes generated runs under `outputs/yolov5/`, which is ignored by Git.

For inference with an existing local checkpoint, place it at `models/best.pt` or set `JTS_YOLO_WEIGHTS` to its location. The CNN notebooks use `JTS_CNN_MODEL` and `JTS_VGG_MODEL` in the same way.

