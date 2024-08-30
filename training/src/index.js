// TODO: creating multiple models using different configs

const tf = require('@tensorflow/tfjs-node');
const {cnn} = require("./model");
const logger = require("./logger");

logger.info('Start training');

const {xTrain, yTrain, xTest, yTest} = require("./data");
const {labelsWithEncoding} = require("./labels");

(async () => {
    const history = await cnn.fit(xTrain, yTrain, {
        epochs: 10,                                                  // Number of epochs
        batchSize: 32,                                              // Number of samples per gradient update
        callbacks: tf.callbacks.earlyStopping({monitor: 'acc', patience: 3})
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

    const summary = yTest.arraySync().map((labels, index) => {
        const isWithSunglasses = labels[0] === labelsWithEncoding.withSunglasses.encoding;

        const label = isWithSunglasses
            ? labelsWithEncoding.withSunglasses.label
            : labelsWithEncoding.withoutSunglasses.label;

        /*
            Every test picture starts at Nr. 661 (not the index)

            Calculation to get the pictures using the index:
            - With Sunglasses:
                - First el : 0 + 661 = 661
                - Last el  : 339 + 661 = 1000
            - Without Sunglasses:
                - First el : 340 + 321 = 661
                - Last el  : 679 + 321 = 1000
         */
        const pictureIndex = isWithSunglasses
            ? Math.floor(index / 2) + 661
            : Math.floor(index / 2) + 321

        return {
            label,
            withSunglassesPrediction: predictedLabels[index][0],
            withoutSunglassesPrediction: predictedLabels[index][1],
            file: `${isWithSunglasses ? 'with-sunglasses' : 'without-sunglasses'}-${pictureIndex}`
        };
    });

    const withSunglassesNegative = summary
        .filter(row => row.label === labelsWithEncoding.withSunglasses.label)
        .filter(row => row.withSunglassesPrediction < row.withoutSunglassesPrediction);

    logger.info('withSunglassesNegative', JSON.stringify(withSunglassesNegative))
    console.table(withSunglassesNegative)

    const withoutSunglassesNegative = summary
        .filter(row => row.label === labelsWithEncoding.withoutSunglasses.label)
        .filter(row => row.withoutSunglassesPrediction < row.withSunglassesPrediction);

    logger.info('withoutSunglassesNegative', JSON.stringify(withoutSunglassesNegative))
    console.table(withoutSunglassesNegative)

    const confusionMatrix = summary.reduce(
        (acc, row) => {
            if (row.label === labelsWithEncoding.withSunglasses.label) {
                row.withSunglassesPrediction > row.withoutSunglassesPrediction
                    ? acc.withSunglassesPositive++
                    : acc.withSunglassesNegative++;
            }

            if (row.label === labelsWithEncoding.withoutSunglasses.label) {
                row.withoutSunglassesPrediction > row.withSunglassesPrediction
                    ? acc.withoutSunglassesPositive++
                    : acc.withoutSunglassesNegative++
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

    logger.info('confusionMatrix', JSON.stringify(confusionMatrix))
    console.table(confusionMatrix)

    await cnn.save('file://model');
})();
