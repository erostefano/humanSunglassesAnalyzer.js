# humanSunglassesAnalyzer.js

This project is designed to analyze images and determine whether a person is wearing sunglasses. This project uses
TensorFlow.js to run machine learning models directly in the browser, so there's no need for a backend server.

## Feature Engineering

* See [Feature Engineering.md](feature-engineering%2FFeature%20Engineering.md)

## Training

* See [Training.md](training%2FTraining.md)

## Application

* Use cam
* Show label

## Operation

* See [Operation.md](Operation%2FOperation.md)

## Thoughts

* Use it inside Google MediaPipe
* What is it really necessary to balance the features in CNN?
* Not reproducible -> Missing Seed
* Code needs unit tests
* e2e Testing
* Store a sidecar to each model including loss, accuracy and confusion matrix
* Data preparation:
  * Brute Force: Train a lot of pictures with different zoom levels etc.
  * Efficient: Locate the eyes with another model and take a screenshot.
* There is no master model -> know your domain and train consequently
