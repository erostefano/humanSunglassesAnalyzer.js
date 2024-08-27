const {loadImagesFromFolder} = require("./util");
const logger = require("./logger");
const tf = require("@tensorflow/tfjs-node");

/**
 * Training data
 */

const withSunglasses = loadImagesFromFolder('../feature-engineering/with-sunglasses', [1, 0]);
const withSunglassesImages = augmentImages(withSunglasses.images);
logger.info('Total with Sunglasses images', withSunglassesImages.length);
const withSunglassesLabels = new Array(withSunglassesImages.length).fill(withSunglasses.label);
logger.info('Total with Sunglasses labels', withSunglassesLabels.length);

const withoutSunglasses = loadImagesFromFolder('../feature-engineering/without-sunglasses', [0, 1]);
const withoutSunglassesImages = augmentImages(withoutSunglasses.images);
logger.info('Total without Sunglasses images', withoutSunglassesImages.length);
const withoutSunglassesLabels = new Array(withoutSunglassesImages.length).fill(withoutSunglasses.label);
logger.info('Total without Sunglasses labels', withoutSunglassesLabels.length);

const splitIndexWithSunglasses = Math.floor(0.66 * withSunglassesImages.length);
logger.info('Split index for with Sunglasses', splitIndexWithSunglasses);

const splitIndexWithoutSunglasses = Math.floor(0.66 * withoutSunglassesImages.length);
logger.info('Split index for without Sunglasses', splitIndexWithoutSunglasses);

const withSunglassesTrainImages = withSunglassesImages.slice(0, splitIndexWithSunglasses);
logger.info('Training images with Sunglasses', withSunglassesTrainImages.length);
const withSunglassesTrainLabels = withSunglassesLabels.slice(0, splitIndexWithSunglasses);
logger.info('Training labels with Sunglasses', withSunglassesTrainLabels.length);

const withoutSunglassesTrainImages = withoutSunglassesImages.slice(0, splitIndexWithoutSunglasses);
logger.info('Training images without Sunglasses', withoutSunglassesTrainImages.length);
const withoutSunglassesTrainLabels = withoutSunglassesLabels.slice(0, splitIndexWithoutSunglasses);
logger.info('Training labels without Sunglasses', withoutSunglassesTrainLabels.length);

const xTrain = tf.stack(withSunglassesTrainImages.concat(withoutSunglassesTrainImages));
logger.info('xTrain shape', xTrain.shape);

const yTrain = tf.tensor2d(withSunglassesTrainLabels.concat(withoutSunglassesTrainLabels), [xTrain.shape[0], 2]);
logger.info('yTrain shape', yTrain.shape);
logger.info('yTrain labels:', yTrain.arraySync());

/**
 * Test data
 */

const withSunglassesTestImages = withSunglassesImages.slice(splitIndexWithSunglasses);
logger.info('Testing images with Sunglasses', withSunglassesTestImages.length);
const withSunglassesTestLabels = withSunglassesLabels.slice(splitIndexWithSunglasses);
logger.info('Testing labels with Sunglasses', withSunglassesTestLabels.length);

const withoutSunglassesTestImages = withoutSunglassesImages.slice(splitIndexWithoutSunglasses);
logger.info('Testing images without Sunglasses', withoutSunglassesTestImages.length);
const withoutSunglassesTestLabels = withoutSunglassesLabels.slice(splitIndexWithoutSunglasses);
logger.info('Testing labels without Sunglasses', withoutSunglassesTestLabels.length);

const xTest = tf.stack(withSunglassesTestImages.concat(withoutSunglassesTestImages));
logger.info('xTest shape', xTest.shape);

const yTest = tf.tensor2d(withSunglassesTestLabels.concat(withoutSunglassesTestLabels), [xTest.shape[0], 2]);
logger.info('yTest shape', yTest.shape);
logger.info('yTest labels:', yTest.arraySync());

module.exports = {xTrain, yTrain, xTest, yTest};

function augmentImages(images) {
    return images
        .map(image => {
            // Add a batch dimension to make the image 4D
            const batchedImage = image.expandDims(0); // Shape becomes [1, height, width, channels]

            // Perform the flip
            const flippedImage = tf.image.flipLeftRight(batchedImage);

            // Remove the batch dimension after flipping
            const unbatchedFlippedImage = flippedImage.squeeze(0); // Shape returns to [height, width, channels]

            return [
                image,                  // Original image
                unbatchedFlippedImage,  // Flipped image
            ];
        })
        .flat(); // Flatten the nested arrays
}
