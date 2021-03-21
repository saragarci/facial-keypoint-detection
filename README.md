[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

This project implements a facial keypoint detection system: given an image, it detects faces and predicts locations of facial keypoints on each face. Facial keypoints include points around the eyes, nose, and mouth on a face.
Applications that use this kind of system are: facial tracking, facial pose recognition, facial filters, and emotion recognition.

![Facial Keypoint Detection][image1]

This project is broken down into four main notebooks:

* Notebook 1: Loading and Visualizing the Facial Keypoint Data
* Notebook 2: Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints
* Notebook 3: Facial Keypoint Detection Using Haar Cascades and your Trained CNN
* Notebook 4: Fun Filters and Keypoint Uses

## Local Environment Instructions

1. Install PyTorch and torchvision; this should install the latest version of PyTorch.
```
pip install torch torchvision torchaudio
```

2. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```

## Credits

### Resources

[Starting project](https://github.com/udacity/P1_Facial_Keypoints). 


### Contributors

* [Sara Garci](s@saragarci.com)
* [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891)

## License

© Copyright 2021 by Sara Garci. All rights reserved.
