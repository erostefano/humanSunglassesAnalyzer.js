const tf = require('@tensorflow/tfjs-node');
const {cnn} = require("./model");
const {loadImagesFromFolder} = require("./util");
const {log} = require("./logger");

const withSunglasses = loadImagesFromFolder('../feature-engineering/with-sunglasses', 1);
const withSunglassesImages = withSunglasses.images;
log('Total with Sunglasses images', withSunglassesImages.length);
const withSunglassesLabels = new Array(withSunglassesImages.length).fill(withSunglasses.label);
log('Total with Sunglasses labels', withSunglassesLabels.length);

const withoutSunglasses = loadImagesFromFolder('../feature-engineering/without-sunglasses', 0);
const withoutSunglassesImages = withoutSunglasses.images;
log('Total without Sunglasses images', withoutSunglassesImages.length);
const withoutSunglassesLabels = new Array(withoutSunglassesImages.length).fill(withoutSunglasses.label);
log('Total without Sunglasses labels', withoutSunglassesLabels.length);

const splitIndexWithSunglasses = Math.floor(0.66 * withSunglassesImages.length);
log('Split index for with Sunglasses', splitIndexWithSunglasses);

const splitIndexWithoutSunglasses = Math.floor(0.66 * withoutSunglassesImages.length);
log('Split index for without Sunglasses', splitIndexWithoutSunglasses);

const withSunglassesTrainImages = withSunglassesImages.slice(0, splitIndexWithSunglasses);
log('Training images with Sunglasses', withSunglassesTrainImages.length);
const withSunglassesTrainLabels = withSunglassesLabels.slice(0, splitIndexWithSunglasses);
log('Training labels with Sunglasses', withSunglassesTrainLabels.length);

const withoutSunglassesTrainImages = withoutSunglassesImages.slice(0, splitIndexWithoutSunglasses);
log('Training images without Sunglasses', withoutSunglassesTrainImages.length);
const withoutSunglassesTrainLabels = withoutSunglassesLabels.slice(0, splitIndexWithoutSunglasses);
log('Training labels without Sunglasses', withoutSunglassesTrainLabels.length);

const xTrain = tf.stack(withSunglassesTrainImages.concat(withoutSunglassesTrainImages));
log('xTrain shape', xTrain.shape);

const yTrain = tf.tensor2d(withSunglassesTrainLabels.concat(withoutSunglassesTrainLabels), [xTrain.shape[0], 1]);
log('yTrain shape', yTrain.shape);

cnn.fit(xTrain, yTrain, {
    epochs: 10,             // Number of epochs
    batchSize: 32,          // Number of samples per gradient update
    callbacks: tf.callbacks.earlyStopping({patience: 3}) // Optional: stops training early if no improvement
})
    .then(history => {
        log('Training completed');
        log('History', JSON.stringify(history));
        log('Final training accuracy', history.history.acc);
        log('Final validation accuracy', history.history.val_acc);

        const withSunglassesTestImages = withSunglassesImages.slice(splitIndexWithSunglasses);
        log('Testing images with Sunglasses', withSunglassesTestImages.length);
        const withSunglassesTestLabels = withSunglassesLabels.slice(splitIndexWithSunglasses);
        log('Testing labels with Sunglasses', withSunglassesTestLabels.length);

        const withoutSunglassesTestImages = withoutSunglassesImages.slice(splitIndexWithoutSunglasses);
        log('Testing images without Sunglasses', withoutSunglassesTestImages.length);
        const withoutSunglassesTestLabels = withoutSunglassesLabels.slice(splitIndexWithoutSunglasses);
        log('Testing labels without Sunglasses', withoutSunglassesTestLabels.length);

        const xTest = tf.stack(withSunglassesTestImages.concat(withoutSunglassesTestImages));
        log('xTest shape', xTest.shape);

        const yTest = tf.tensor2d(withSunglassesTestLabels.concat(withoutSunglassesTestLabels), [xTest.shape[0], 1]);
        log('yTest shape', yTest.shape);

        return cnn.evaluate(xTest, yTest);
    })
    .then(testResult => {
        log('Testresult', JSON.stringify(testResult))
        const [lossTensor, accuracyTensor] = testResult;

        log('Test Loss', lossTensor.dataSync());
        log('Test Accuracy', accuracyTensor.dataSync());

        return cnn.save('file://model');
    })
    .then(() => {
        log('Model saved to disk');
    })
    .catch(error => {
        error('Error during training, evaluation, or saving', error);
    });
