// TODO: create a confusion matrix using .predict and the test data
// TODO: use async
// TODO: creating multiple models using different configs

const tf = require('@tensorflow/tfjs-node');
const {cnn} = require("./model");
const logger = require("./logger");

logger.info('Start training');

const {xTrain, yTrain, xTest, yTest} = require("./data");

cnn.fit(xTrain, yTrain, {
    epochs: 10,             // Number of epochs
    batchSize: 32,          // Number of samples per gradient update
    callbacks: tf.callbacks.earlyStopping({patience: 3}) // Optional: stops training early if no improvement
})
    .then(history => {
        logger.info('Training completed');
        logger.info('History', JSON.stringify(history));
        logger.info('Final training accuracy', history.history.acc);
        logger.info('Final validation accuracy', history.history.val_acc);

        return cnn.evaluate(xTest, yTest);
    })
    .then(testResult => {
        logger.info('Testresult', JSON.stringify(testResult))
        const [lossTensor, accuracyTensor] = testResult;

        logger.info('Test Loss', lossTensor.dataSync());
        logger.info('Test Accuracy', accuracyTensor.dataSync());

        return cnn.save('file://model');
    })
    .then(() => {
        logger.info('Model saved to disk');
    })
    .catch(error => {
        logger.error('Error during training, evaluation, or saving', error);
    });
