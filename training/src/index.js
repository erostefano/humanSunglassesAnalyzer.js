// Import TensorFlow.js
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Create a sequential model
const model = tf.sequential();

// Add the first convolutional layer
model.add(tf.layers.conv2d({
    inputShape: [224, 224, 3],  // Adjusted for 224x224 RGB images
    filters: 32,                // Number of filters
    kernelSize: 3,              // 3x3 filter size
    activation: 'relu'          // ReLU activation function
}));

// Add a max-pooling layer
model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

// Add the second convolutional layer
model.add(tf.layers.conv2d({
    filters: 64,                // Increase filters to capture more complex features
    kernelSize: 3,
    activation: 'relu'
}));

// Add a second max-pooling layer
model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

// Add a third convolutional layer
model.add(tf.layers.conv2d({
    filters: 128,               // Further increase filters
    kernelSize: 3,
    activation: 'relu'
}));

// Add a third max-pooling layer
model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

// Add a fourth convolutional layer (optional for larger images)
model.add(tf.layers.conv2d({
    filters: 256,               // Further increase filters
    kernelSize: 3,
    activation: 'relu'
}));

// Add a fourth max-pooling layer
model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

// Flatten the output to feed it into a fully connected layer
model.add(tf.layers.flatten());

// Add a fully connected (dense) layer
model.add(tf.layers.dense({
    units: 256,                 // Increase number of neurons for larger input
    activation: 'relu'
}));

// Optionally add a dropout layer to reduce overfitting
//model.add(tf.layers.dropout({rate: 0.5}));

// Add the output layer with a single neuron (binary classification)
model.add(tf.layers.dense({
    units: 1,                   // Output unit
    activation: 'sigmoid'       // Sigmoid activation for binary classification
}));

// Compile the model with a binary cross-entropy loss and an optimizer
model.compile({
    optimizer: tf.train.adam(),        // Adam optimizer
    loss: 'binaryCrossentropy',        // Binary Cross-Entropy loss
    metrics: ['accuracy']              // Track accuracy
});


// Function to load and preprocess a single image
function loadImage(filePath) {
    const buffer = fs.readFileSync(filePath);// Read the image file
    const imageTensor = tf.node.decodeImage(buffer, 3); // Decode the image to a tensor
    const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]); // Resize to 224x224
    const normalizedImage = resizedImage.div(255.0); // Normalize pixel values to [0, 1]
    return normalizedImage;
}

// Function to load images from a folder and assign the same label
function loadImagesFromFolder(folderPath, label) {
    const files = fs.readdirSync(folderPath); // Get all files in the folder
    const images = files.map(file => loadImage(path.join(folderPath, file))); // Load and preprocess each image
    return {images, label};
}

// Load images from both folders
const withSunglasses = loadImagesFromFolder('../feature-engineering/with-sunglasses', 1);
const withoutSunglasses = loadImagesFromFolder('../feature-engineering/without-sunglasses', 0);

// Ensure that images and labels are loaded correctly
if (!withSunglasses || !withoutSunglasses) {
    throw new Error("Failed to load images from one or both folders.");
}

const withSunglassesImages = withSunglasses.images;
const withSunglassesLabels = new Array(withSunglassesImages.length).fill(withSunglasses.label);

const withoutSunglassesImages = withoutSunglasses.images;
const withoutSunglassesLabels = new Array(withoutSunglassesImages.length).fill(withoutSunglasses.label);

// Log the size of each feature before splitting
console.log('Total with Sunglasses images:', withSunglassesImages.length);
console.log('Total with Sunglasses labels:', withSunglassesLabels.length);
console.log('Total without Sunglasses images:', withoutSunglassesImages.length);
console.log('Total without Sunglasses labels:', withoutSunglassesLabels.length);

// Calculate split index for 66% training data
const splitIndexWithSunglasses = Math.floor(0.66 * withSunglassesImages.length);
const splitIndexWithoutSunglasses = Math.floor(0.66 * withoutSunglassesImages.length);

// Log split indices
console.log('Split index for with Sunglasses:', splitIndexWithSunglasses);
console.log('Split index for without Sunglasses:', splitIndexWithoutSunglasses);

// Split images and labels for each class
const withSunglassesTrainImages = withSunglassesImages.slice(0, splitIndexWithSunglasses);
const withSunglassesTrainLabels = withSunglassesLabels.slice(0, splitIndexWithSunglasses);

const withSunglassesTestImages = withSunglassesImages.slice(splitIndexWithSunglasses);
const withSunglassesTestLabels = withSunglassesLabels.slice(splitIndexWithSunglasses);

const withoutSunglassesTrainImages = withoutSunglassesImages.slice(0, splitIndexWithoutSunglasses);
const withoutSunglassesTrainLabels = withoutSunglassesLabels.slice(0, splitIndexWithoutSunglasses);

const withoutSunglassesTestImages = withoutSunglassesImages.slice(splitIndexWithoutSunglasses);
const withoutSunglassesTestLabels = withoutSunglassesLabels.slice(splitIndexWithoutSunglasses);

// Log the size of each feature after splitting
console.log('Training images with Sunglasses:', withSunglassesTrainImages.length);
console.log('Training labels with Sunglasses:', withSunglassesTrainLabels.length);
console.log('Testing images with Sunglasses:', withSunglassesTestImages.length);
console.log('Testing labels with Sunglasses:', withSunglassesTestLabels.length);

console.log('Training images without Sunglasses:', withoutSunglassesTrainImages.length);
console.log('Training labels without Sunglasses:', withoutSunglassesTrainLabels.length);
console.log('Testing images without Sunglasses:', withoutSunglassesTestImages.length);
console.log('Testing labels without Sunglasses:', withoutSunglassesTestLabels.length);

// Combine training data and labels
const xTrain = tf.stack(withSunglassesTrainImages.concat(withoutSunglassesTrainImages));
const yTrain = tf.tensor2d(withSunglassesTrainLabels.concat(withoutSunglassesTrainLabels), [xTrain.shape[0], 1]);

// Combine testing data and labels
const xTest = tf.stack(withSunglassesTestImages.concat(withoutSunglassesTestImages));
const yTest = tf.tensor2d(withSunglassesTestLabels.concat(withoutSunglassesTestLabels), [xTest.shape[0], 1]);

// Confirm the shapes of your tensors
console.log('xTrain shape:', xTrain.shape);
console.log('yTrain shape:', yTrain.shape);
console.log('xTest shape:', xTest.shape);
console.log('yTest shape:', yTest.shape);

model.fit(xTrain, yTrain, {
    epochs: 10,             // Number of epochs
    batchSize: 32,          // Number of samples per gradient update
    callbacks: tf.callbacks.earlyStopping({patience: 3}) // Optional: stops training early if no improvement
})
    .then(info => {
        console.log('Training Complete');
        console.log('Final training accuracy:', info.history.acc);
        console.log('Final validation accuracy:', info.history.val_acc);

        // After training is complete, evaluate the model
        return model.evaluate(xTest, yTest);
    })
    .then(testResult => {
        // If you have multiple metrics, testResult will be an array of tensors
        // Example: [lossTensor, accuracyTensor]
        const [lossTensor, accuracyTensor] = testResult;

        // Convert tensors to arrays and print the results
        console.log('Test Loss:', lossTensor.dataSync());
        console.log('Test Accuracy:', accuracyTensor.dataSync());

        // Save the model after evaluation
        return model.save('file://model');
    })
    .then(() => {
        console.log('Model saved to disk');
    })
    .catch(error => {
        console.error('Error during training, evaluation, or saving:', error);
    });
