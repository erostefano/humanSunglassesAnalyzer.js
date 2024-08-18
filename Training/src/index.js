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
const withSunglasses = loadImagesFromFolder('../Feature_Engineering/with_sunglasses', 1);
const withoutSunglasses = loadImagesFromFolder('../Feature_Engineering/without_sunglasses', 0);

const allImages = withSunglasses.images.concat(withoutSunglasses.images);

const allLabels = tf.tensor2d(
    new Array(withSunglasses.images.length).fill(withSunglasses.label)
        .concat(new Array(withoutSunglasses.images.length).fill(withoutSunglasses.label)),
    [allImages.length, 1]
);

// Convert the list of images into a 4D tensor
const xData = tf.stack(allImages);

// Calculate the split index for 66% training and 33% testing
const trainSize = Math.floor(0.66 * xData.shape[0]);

// Split the data into training and testing sets
const xTrain = xData.slice([0, 0, 0, 0], [trainSize, 224, 224, 3]);
const yTrain = allLabels.slice([0, 0], [trainSize, 1]);

// Train the model
model.fit(xTrain, yTrain, {
    epochs: 10,             // Number of epochs
    batchSize: 32,          // Number of samples per gradient update
    callbacks: tf.callbacks.earlyStopping({patience: 3}) // Optional: stops training early if no improvement
})
    .then(info => {
        console.log('Training Complete');
        console.log('Final training accuracy:', info.history.acc);
        console.log('Final validation accuracy:', info.history.val_acc);
    })
    .catch(error => {
        console.log(error);
    });

const xTest = xData.slice([trainSize, 0, 0, 0], [xData.shape[0] - trainSize, 224, 224, 3]);
const yTest = allLabels.slice([trainSize, 0], [allLabels.shape[0] - trainSize, 1]);

model.evaluate(xTest, yTest)
    .then(testResult => {
        // If you have multiple metrics, testResult will be an array of tensors
        // Example: [lossTensor, accuracyTensor]
        const [lossTensor, accuracyTensor] = testResult;

        // Print the loss and accuracy values
        console.log('Test Loss:', lossTensor.dataSync()); // Convert tensor to array and log
        console.log('Test Accuracy:', accuracyTensor.dataSync()); // Convert tensor to array and log
    })
    .catch(error => {
        console.error('Error during evaluation:', error);
    });

// After training the model, save it to the filesystem
model.save('model')
    .then(() => {
        console.log('Model saved to disk');
    })
    .catch(error => {
        console.error('Error saving the model:', error);
    });
