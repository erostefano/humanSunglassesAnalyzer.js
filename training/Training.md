# Training

The model is trained in a Node.js environment using the TensorFlow.js library.

## Convolutional Neural Network

TODO: mention preprocessing

TODO: mention data augmentation

TODO: brief cnn architecture, mention missing seed

### Output Units

In the initial version, the model used a single output unit that produced a value between 0 and 1. A value closer to 1
meant sunglasses were more likely present, while a value closer to 0 meant sunglasses were less likely present. For
example:

```
Image 1: 0.8 = with sunglasses
Image 2: 0.2 = without sunglasses
```

In the final version, the model was updated to use two output units, which clearly indicate the predictions for each
label:

```
Image 1: 0.8 = with sunglasses, 0.2 = without sunglasses
```

TODO: mention param tuning

TODO: add training/test loss and accuracy, confusion matrix and false negatives, sum up everything

TODO: write about storing models including sidecars container loss, accuracy and commit id for images

TODO: model is trained on close ups. it doesnt work nice on webcam. potential fixes:
- train it more
- fix it in the frontend using a mask and cutting the picture
- use face-api model to get the face and predict it
- move this model into a media pipe
