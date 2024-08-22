const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

function loadImage(filePath) {
    const buffer = fs.readFileSync(filePath);// Read the image file
    const imageTensor = tf.node.decodeImage(buffer, 3); // Decode the image to a tensor
    const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]); // Resize to 224x224
    return resizedImage.div(255.0);
}

function loadImagesFromFolder(folderPath, label) {
    const files = fs.readdirSync(folderPath); // Get all files in the folder
    const images = files.map(file => loadImage(path.join(folderPath, file))); // Load and preprocess each image
    return {images, label};
}

module.exports = {loadImagesFromFolder}