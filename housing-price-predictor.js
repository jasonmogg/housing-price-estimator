import csv from 'csvtojson';
import fs from 'node:fs';
import pkg from 'ml-random-forest';
const { RandomForestRegression } = pkg;

export class HousingPricePredictor {
    // Features used for prediction - can be modified based on your dataset
    #features = Object.freeze(['bedrooms', 'bathrooms', 'yearBuilt', 'livingArea', 'lotSize']);
    #regression;
    #trainingPromise;
    #options;

    constructor(trainingPath, modelPath, options = {
        maxFeatures: 5, // defaults to number of features
        nEstimators: 100, // number of trees
        replacement: false,
        seed: 42
    }) {
        this.#options = options;

        if (modelPath && fs.existsSync(modelPath)) {
            console.log('Loading model from', modelPath);

            this.#trainingPromise = new Promise(resolve => {
                const modelData = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
                this.#regression = RandomForestRegression.load(modelData);
                console.log('Model loaded successfully');
                resolve(this.#regression);
            });
        } else {
            console.log('Training model from', trainingPath);

            this.#trainingPromise = this.#load(trainingPath).then(({ data, headers, prices }) => {
                console.log(`Training data loaded: ${data.length} records with features: ${this.#features.join(', ')}`);
                
                // Create and train the random forest regressor
                this.#regression = new RandomForestRegression(this.#options);
                this.#regression.train(data, prices);
                
                console.log('Model training complete');
                
                if (modelPath) {
                    fs.writeFileSync(modelPath, JSON.stringify(this.#regression.toJSON()));
                    console.log('Model saved to', modelPath);
                }
                
                return this.#regression;
            });
        }
    }

    // Load and process data from CSV file
    #load(path) {
        return csv().fromFile(path).then(rows => {
            const headers = Object.keys(rows[0] || {});
            
            // Filter to only include valid numerical features
            const validRows = rows.filter(row => {
                return this.#features.every(feature => 
                    headers.includes(feature) && 
                    !isNaN(parseFloat(row[feature]))
                ) && 
                headers.includes('price') && 
                !isNaN(parseFloat(row.price));
            });
            
            if (validRows.length === 0) {
                throw new Error('No valid data found in CSV file');
            }

            // Extract features and prices
            const data = validRows.map(row => 
                this.#features.map(feature => parseFloat(row[feature]))
            );
            
            const prices = validRows.map(row => parseFloat(row.price));
            
            return {
                data,
                headers,
                prices
            };
        });
    }

    // Predict prices for new data
    predict(inputPath) {
        return Promise.all([
            this.#trainingPromise,
            this.#load(inputPath)
        ]).then(([, { data, headers }]) => {
            if (!this.#regression) {
                throw new Error('Model not trained or loaded');
            }
            
            // Make predictions
            const predictions = this.#regression.predict(data);
            
            return {
                data,
                predictions,
                headers
            };
        });
    }

    // Calculate evaluation metrics if actual prices are available
    evaluate(inputPath) {
        return Promise.all([
            this.#trainingPromise,
            this.#load(inputPath)
        ]).then(([, { data, prices }]) => {
            if (!this.#regression) {
                throw new Error('Model not trained or loaded');
            }
            
            // Make predictions
            const predictions = this.#regression.predict(data);
            
            // Calculate metrics
            const metrics = this.#calculateMetrics(predictions, prices);
            
            return {
                data,
                predictions,
                metrics
            };
        });
    }

    // Calculate common regression metrics
    #calculateMetrics(predictions, actuals) {
        if (predictions.length !== actuals.length) {
            throw new Error('Predictions and actuals must have the same length');
        }
        
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
        
        return { mae, mse, rmse, r2 };
    }
}

export default HousingPricePredictor;
