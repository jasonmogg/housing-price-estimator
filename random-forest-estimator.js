import csv from 'csvtojson';
import fs from 'node:fs';
import { RandomForestRegression } from 'ml-random-forest';

export class RandomForestEstimator {
    #regression;
    #trainingPromise;

    constructor (trainingPath, modelPath, options = {
        maxFeatures: 1.0,
        nEstimators: 200,
        replacement: false,
        seed: 42
    }) {
        if (modelPath && fs.existsSync(modelPath)) {
            console.log('Loading model from', modelPath);

            this.#trainingPromise = new Promise(async resolve => {
                this.#regression = RandomForestRegression.load(JSON.parse(await fs.promises.readFile(modelPath)));

                resolve(this.#regression);
            });
        } else {
            console.log('Training model from', trainingPath);

            this.#trainingPromise = this.#load(trainingPath).then(({ actuals, data, rows }) => {
                const labels = rows[0];

                console.log('Training data loaded:');

                data.forEach((row, index) => console.log(row, actuals[index]));

                this.#regression = new RandomForestRegression(options);
                this.#regression.train(data, actuals);

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

    // public properties

    get featureFilters () {
        return Object.freeze({});
    }

    get categoricalValues () {
        return Object.freeze({});
    }

    get features () {
        throw new Error('Not implemented');
    }

    get predictionField () {
        throw new Error('Not implemented');
    }

    // methods

    actualFilter (actual) {
        return true;
    }

    // Calculate common regression metrics
    static #calculateMetrics(predictions, actuals) {
        if (predictions.length !== actuals.length) {
            throw new Error('Predictions and actuals must have the same length');
        }

        const diffs = predictions.map((pred, i) => (pred / actuals[i]) - 1);
        
        // Calculate Mean Absolute Error (MAE)
        const mae = predictions.reduce((sum, pred, i) => 
            sum + Math.abs(pred - actuals[i]), 0) / predictions.length;
            
        // Calculate Mean Squared Error (MSE)
        const mse = predictions.reduce((sum, pred, i) => 
            sum + Math.pow(pred - actuals[i], 2), 0) / predictions.length;
            
        // Calculate Root Mean Squared Error (RMSE)
        const rmse = Math.sqrt(mse);
        
        // Calculate R-squared
        const mean = actuals.reduce((sum, val) => sum + val, 0) / actuals.length;
        const totalVariance = actuals.reduce((sum, val) => 
            sum + Math.pow(val - mean, 2), 0);
        const residualVariance = predictions.reduce((sum, pred, i) => 
            sum + Math.pow(actuals[i] - pred, 2), 0);
        const r2 = 1 - (residualVariance / totalVariance);
        
        return { diffs, mae, mse, rmse, r2 };
    }

    #load (path) {
        return csv({ noheader: true, output: 'csv' }).fromFile(path).then(rows => {
            const actuals = [];
            const labels = rows[0];
            const dataRows = rows.slice(1);
            const data = [];
            const predictionIndex = labels.indexOf(this.predictionField);
            
            dataRows.forEach(row => {
                const actual = Number(row[predictionIndex]);

                if (!this.actualFilter(actual)) {
                    return;
                }

                const rowData = [];

                for (let i = 0; i < this.features.length; i++) {
                    const feature = this.features[i];
                    let value = row[labels.indexOf(feature)];

                    if (this.categoricalValues[feature]) {
                        value = this.categoricalValues[feature].indexOf(value);

                        if (value === -1) {
                            console.log('Missing categorical value:', row[labels.indexOf(feature)]);

                            return;
                        }
                    } else {
                        value = Number(value);
                    }

                    if (this.featureFilters[feature] && !this.featureFilters[feature](value)) {
                        return;
                    }

                    rowData.push(value);
                }

                actuals.push(actual);
                data.push(rowData);
            });

            return {
                actuals,
                data,
                rows
            };
        });
    }

    // Calculate evaluation metrics if actuals are available
    run(inputPath) {
        return Promise.all([
            this.#trainingPromise,
            this.#load(inputPath)
        ]).then(([, { actuals, data }]) => {
            if (!this.#regression) {
                throw new Error('Model not trained or loaded');
            }
            
            // Make predictions
            const predictions = this.#regression.predict(data);
            
            let metrics = null;

            if (actuals) {
                // Calculate metrics
                metrics = RandomForestEstimator.#calculateMetrics(predictions, actuals);
            }
            
            return {
                data,
                metrics,
                predictions,
                actuals
            };
        });
    }
}

export default RandomForestEstimator;
