// TODO: create a confusion matrix using .predict and the test data
// TODO: creating multiple models using different configs

const tf = require('@tensorflow/tfjs-node');
const {cnn} = require("./model");
const logger = require("./logger");

logger.info('Start training');

const {xTrain, yTrain, xTest, yTest} = require("./data");

(async () => {
    const history = await cnn.fit(xTrain, yTrain, {
        epochs: 10,             // Number of epochs
        batchSize: 32,          // Number of samples per gradient update
        callbacks: tf.callbacks.earlyStopping({patience: 3}) // Optional: stops training early if no improvement
    });

    logger.info('Training completed');
    logger.info('History', JSON.stringify(history));
    logger.info('Training loss', history.history.loss);
    logger.info('Training accuracy', history.history.acc);

    const testResult = cnn.evaluate(xTest, yTest);
    logger.info('Test result', JSON.stringify(testResult))

    const [lossTensor, accuracyTensor] = testResult;
    logger.info('Test Loss', lossTensor.dataSync());
    logger.info('Test Accuracy', accuracyTensor.dataSync());

    await cnn.save('file://model');
})();
