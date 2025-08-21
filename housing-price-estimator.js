import csv from 'csvtojson';
import fs from 'node:fs';
import { RandomForestClassifier } from 'ml-random-forest';

const PRICE_LABEL = 'price';

export class HousingPriceEstimator {
    #features = Object.freeze(['bedrooms', 'bathrooms', 'yearBuilt', 'livingArea', 'lotSize']);
    #regression;
    #trainingPromise;

    constructor (trainingPath, modelPath, options = {
        maxFeatures: this.#features.length,
        nEstimators: 200,
        replacement: false,
        seed: 42
    }) {
        if (modelPath && fs.existsSync(modelPath)) {
            console.log('Loading model from', modelPath);

            this.#trainingPromise = new Promise(resolve => {
                this.#regression = RandomForestClassifier.load(JSON.parse(fs.readSync(modelPath)));

                resolve(this.#regression);
            });
        } else {
            console.log('Training model from', trainingPath);

            this.#trainingPromise = this.#load(trainingPath).then(({ data, rows }) => {
                const labels = rows[0];

                const priceData = rows.slice(1).map(row => row[labels.indexOf(PRICE_LABEL)]);

                console.log('Training data loaded:');

                data.forEach((row, index) => console.log(row, priceData[index]));

                this.#regression = new RandomForestClassifier(options);
                this.#regression.train(data, priceData);

                if (modelPath) {
                    fs.writeFileSync(modelPath, JSON.stringify(this.#regression.toJSON()));

                    console.log('Model saved to', modelPath);
                }

                return this.#regression;
            });
        }

        this.#trainingPromise.then(regression => {
            console.log(regression.featureImportance());
        });
    }

    #load (path) {
        return csv({ noheader: true, output: 'csv' }).fromFile(path).then(rows => {
            const labels = rows[0];

            return {
                data: rows.slice(1).map(row => this.#features.map(feature => Number(row[labels.indexOf(feature)]))),
                rows
            };
        });
    }

    run (inputPath) {
        const inputPromise = this.#load(inputPath);

        return Promise.all([
            this.#trainingPromise,
            inputPromise,
        ]).then(([, { data: inputData }]) => {
            return {
                data: inputData,
                predictions: this.#regression.predict(inputData)
            };
        });
    }
}

export default HousingPriceEstimator;
