# Dataset setup

The dataset is not included in this repository. Download it from the official Kaggle page:

- [Jordanian (Arabic) Traffic Signs in YOLO Format](https://www.kaggle.com/datasets/khaledhweij/jordanian-traffic-signs)

After extraction, the default expected layout is:

```text
data/
└── Final_Dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

The original exploratory notebook recorded 6,923 training images, 1,623 validation images, and 700 testing images. Set `JTS_DATASET_ROOT` if you store `Final_Dataset` elsewhere.

Do not place `kaggle.json` in this repository. Use the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables, or keep Kaggle's configuration file only in its standard user-level directory.

The dataset remains governed by the license and terms shown on its Kaggle page. Review the current terms before use or redistribution.

