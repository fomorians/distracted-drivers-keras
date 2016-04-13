# Distracted Drivers Starter Project

This starter project currently ranks in the top 10% of submissions.

Get started using AWS spot instances without the hassle of setting up infrastructure.

## Dataset Preparation

1. Download the images and drivers list into the "dataset" folder.
2. Unzip both.
3. Run the `prep_dataset.py` script.

## Training

Before using this code you must agree to the [Kaggle competition's terms of use](https://www.kaggle.com/c/state-farm-distracted-driver-detection).

### Cloud Setup

1. Follow the [installation guide](https://fomoro.gitbooks.io/guide/content/installation.html) for [Fomoro](https://fomoro.com). [Reach out](mailto:team@fomoro.com) to us if you need help getting started.
2. Clone the repo: `git clone https://github.com/fomorians/distracted-drivers.git && cd distracted-drivers`
3. Create a new model: `fomoro model add`
4. Start training: `fomoro session start -f`

### Local Setup

1. [Install TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation).
2. Clone the repo: `git clone https://github.com/fomorians/distracted-drivers.git && cd distracted-drivers`
3. Run training: `python main.py`

## Model Development

1. Tune your learning rate. If the loss diverges, the learning rate is probably too high. If it never learns, it might be too low.
2. Good initialization is important. The initial values of weights can have a significant impact on learning. In general, you want the weights initialized based on the input and/or output dimensions (see Glorot or He initialization).
