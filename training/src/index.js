// TODO: replace logger
// TODO: create a confusion matrix using .predict and the test data
// TODO: use async
// TODO: creating multiple models using different configs

const tf = require('@tensorflow/tfjs-node');
const {cnn} = require("./model");
const {log} = require("./logger");
const {xTrain, yTrain, xTest, yTest} = require("./data");

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
