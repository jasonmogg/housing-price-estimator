import csv from 'csvtojson';
import fs from 'node:fs';
import { RandomForestRegression } from 'ml-random-forest';

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
                this.#regression = RandomForestRegression.load(JSON.parse(fs.readSync(modelPath)));

                resolve(this.#regression);
            });
        } else {
            console.log('Training model from', trainingPath);

            this.#trainingPromise = this.#load(trainingPath).then(({ data, prices, rows }) => {
                const labels = rows[0];

                console.log('Training data loaded:');

                data.forEach((row, index) => console.log(row, prices[index]));

                this.#regression = new RandomForestRegression(options);
                this.#regression.train(data, prices);

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

    // Calculate common regression metrics
    #calculateMetrics(predictions, actuals) {
        if (predictions.length !== actuals.length) {
            throw new Error('Predictions and actuals must have the same length');
        }

        const diffs = predictions.map((pred, i) => (pred - actuals[i]) / actuals[i]);
        
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
            const labels = rows[0];

            const dataRows = rows.slice(1);

            return {
                data: dataRows.map(row => this.#features.map(feature => Number(row[labels.indexOf(feature)]))),
                prices: labels.indexOf(PRICE_LABEL) === -1 ? null : dataRows.map(row => Number(row[labels.indexOf(PRICE_LABEL)])),
                rows
            };
        });
    }

    // Calculate evaluation metrics if actual prices are available
    run(inputPath) {
        return Promise.all([
            this.#trainingPromise,
            this.#load(inputPath)
        ]).then(([, { data, prices }]) => {
            if (!this.#regression) {
                throw new Error('Model not trained or loaded');
            }
            
            // Make predictions
            const predictions = this.#regression.predict(data);
            
            let metrics = null;

            if (prices) {
                // Calculate metrics
                metrics = this.#calculateMetrics(predictions, prices);
            }
            
            return {
                data,
                metrics,
                predictions,
                prices
            };
        });
    }
}

export default HousingPriceEstimator;
