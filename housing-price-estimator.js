import csv from 'csvtojson';
import { RandomForestRegression } from 'ml-random-forest';

const PRICE_LABEL = 'price';

class HousingPriceEstimator {
    #features = Object.freeze(['bedrooms', 'bathrooms', 'yearBuilt', 'livingArea', 'lotSize']);
    #regression;
    #trainingPromise;

    constructor (trainingPath, options = {
        maxFeatures: this.#features.length,
        nEstimators: 200,
        replacement: false,
        seed: 42
    }) {
        this.#trainingPromise = this.#load(trainingPath).then(({ data, rows }) => {
            const labels = rows[0];

            const priceData = rows.slice(1).map(row => row[labels.indexOf(PRICE_LABEL)]);

            data.forEach((row, index) => console.log(row, priceData[index]));

            this.#regression = new RandomForestRegression(options);
            this.#regression.train(data, priceData);

            console.log(this.#regression.featureImportance());

            return data;
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
        ]).then(([data, { data: inputData }]) => {
            return {
                data,
                predictions: this.#regression.predict(inputData)
            };
        });
    }
}

new HousingPriceEstimator(process.argv[2]).run(process.argv[3]).then(({ data, predictions }) => {
    data.forEach((row, index) => console.log(row, predictions[index]));
});
