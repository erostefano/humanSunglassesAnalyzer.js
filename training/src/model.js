const tf = require('@tensorflow/tfjs-node');

const cnn = tf.sequential();

// Add the first convolutional layer
cnn.add(tf.layers.conv2d({
    inputShape: [224, 224, 3],  // Adjusted for 224x224 RGB images
    filters: 32,                // Number of filters
    kernelSize: 3,              // 3x3 filter size
    activation: 'relu'          // ReLU activation function
}));

// Add a max-pooling layer
cnn.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

// Add the second convolutional layer
cnn.add(tf.layers.conv2d({
    filters: 64,                // Increase filters to capture more complex features
    kernelSize: 3,
    activation: 'relu'
}));

// Add a second max-pooling layer
cnn.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

// Add a third convolutional layer
cnn.add(tf.layers.conv2d({
    filters: 128,               // Further increase filters
    kernelSize: 3,
    activation: 'relu'
}));

// Add a third max-pooling layer
cnn.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

// Add a fourth convolutional layer (optional for larger images)
cnn.add(tf.layers.conv2d({
    filters: 256,               // Further increase filters
    kernelSize: 3,
    activation: 'relu'
}));

// Add a fourth max-pooling layer
cnn.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

// Flatten the output to feed it into a fully connected layer
cnn.add(tf.layers.flatten());

// Add a fully connected (dense) layer
cnn.add(tf.layers.dense({
    units: 256,                 // Increase number of neurons for larger input
    activation: 'relu'
}));

// Optionally add a dropout layer to reduce overfitting
//cnn.add(tf.layers.dropout({rate: 0.5}));

// Add the output layer with a single neuron (binary classification)
cnn.add(tf.layers.dense({
    units: 1,                   // Output unit
    activation: 'sigmoid'       // Sigmoid activation for binary classification
}));

// Compile the cnn with a binary cross-entropy loss and an optimizer
cnn.compile({
    optimizer: tf.train.adam(),        // Adam optimizer
    loss: 'binaryCrossentropy',        // Binary Cross-Entropy loss
    metrics: ['accuracy']              // Track accuracy
});

module.exports = {cnn};
