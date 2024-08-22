// TODO: creating multiple models using different configs

const tf = require('@tensorflow/tfjs-node');
const {cnn} = require("./model");
const logger = require("./logger");

logger.info('Start training');

const {xTrain, yTrain, xTest, yTest} = require("./data");

(async () => {
    const history = await cnn.fit(xTrain, yTrain, {
        epochs: 1,                                                  // Number of epochs
        batchSize: 32,                                              // Number of samples per gradient update
        callbacks: tf.callbacks.earlyStopping({patience: 3})   // Optional: stops training early if no improvement
    });

    logger.info('Training completed');
    logger.info('History', JSON.stringify(history));
    logger.info('Training loss', history.history.loss);
    logger.info('Training accuracy', history.history.acc);

    const testResult = cnn.evaluate(xTest, yTest);
    logger.info('Test result', JSON.stringify(testResult))

    const [lossTensor, accuracyTensor] = testResult;
    logger.info('Test loss', lossTensor.dataSync());
    logger.info('Test accuracy', accuracyTensor.dataSync());

    const predictions = cnn.predict(xTest);
    const predictedLabels = predictions.arraySync();

    logger.info('Predicted Labels', predictedLabels)
    logger.info('Predicted Labels Size', predictedLabels.length)
    logger.info('Expected Labels', yTest.arraySync())
    logger.info('Expected Labels', yTest.arraySync().length)

    const table = yTest.arraySync().map((labels, index) => {
        const label = labels[0];
        const prediction = predictedLabels[index][0];
        return {
            label,
            prediction
        };
    });

    const confusionMatrix = table.reduce(
        (acc, {label, prediction}) => {
            if (label === 1) {
                prediction >= 0.5 ? acc.withSunglassesPositive++ : acc.withSunglassesNegative++;
            }

            if (label === 0) {
                prediction < 0.5 ? acc.withoutSunglassesPositive++ : acc.withoutSunglassesNegative++;
            }

            return acc;
        },
        {
            withSunglassesPositive: 0,
            withSunglassesNegative: 0,
            withoutSunglassesPositive: 0,
            withoutSunglassesNegative: 0
        }
    );

    console.table(confusionMatrix)

    await cnn.save('file://model');
})();
