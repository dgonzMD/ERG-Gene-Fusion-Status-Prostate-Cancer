# ERG-Gene-Fusion-Status-Prostate-Cancer

## About
Supplemental code for "Leveraging Artificial Intelligence to Predict ERG Gene Fusion Status in Prostate Cancer"

Regions of tumor from WSIs were manually annotated using QuPath v0.2.3. The regions annotated as either ERG-positive or ERG-negative were exported as 224x224 pixel sized JPEG image patches at 10x, 20x, and 40x magnifications, from the same regions of interest, for input into the deep learning model. 

235 TCGA cases and 26 in-house cases were used for our training.

Using the Python Keras Application Programming Interface (API), we developed a deep learning algorithm for distinguishing between ERG rearranged and ERG non-rearranged prostate cancer. 

## Pre-requisites

Development and testing was performed using a computer equipped with an NVIDIA RTX 2070 Super graphics processing unit (GPU) and 16GB of RAM at 3200MHz. The algorithm is based on the MobileNetV2 convolutional neural network (CNN) architecture pre-trained on ImageNet.

Python (3.7), TensorFlow(2.1), matplotlib (3.3.4), Keras(2.4.3), numpy(1.19.5), scikit-learn(0.24.2), pandas(1.2.4)

## Model Training and Evaluation

To train a model using our code, run the model_training.py file:

``` shell
python model_training.py
```

Select the desired dataset to be trained on. The dataset is expected to have the following directory format:

```bash
DATA_DIRECTORY/
	├── ERG_Positive
        ├── patch 1.jpeg
        ├── patch 2.jpeg
        └── ...
	├── ERG_Negative
        ├── patch 3.jpeg
        ├── patch 4.jpeg
	└── ...
```

To evaluate the model, run evaluate.py:

``` shell
python evaluate.py
```

Select the desired dataset to be evaluated. The dataset is expected to have the following directory format:

```bash
DATA_DIRECTORY/
	├──10x
      ├── ERG_Positive
            ├── patch 1.jpeg
            ├── patch 2.jpeg
            └── ...
      ├── ERG_Negative
            ├── patch 3.jpeg
            ├── patch 4.jpeg
  ├──20x
      ├── ERG_Positive
            ├── patch 5.jpeg
            ├── patch 6.jpeg
            └── ...
      ├── ERG_Negative
            ├── patch 7.jpeg
            ├── patch 8.jpeg
  ├──40x
      ├── ERG_Positive
            ├── patch 9.jpeg
            ├── patch 10.jpeg
            └── ...
      ├── ERG_Negative
            ├── patch 11.jpeg
            ├── patch 12.jpeg
	└── ...
```

Where the ERG_Positive and ERG_Negative folders represent the *true* labels for the respective image patches. This allows the model to evaluate the patches and assign correctness for statistical metrics.

## Issues
- Please report all issues on the public forum.
