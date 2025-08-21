#!/usr/bin/env node

import { HousingPriceEstimator } from './housing-price-estimator.js';
import fs from 'node:fs';

// Show usage if insufficient arguments
if (process.argv.length < 3) {
    console.error('Usage: node index.js <training_csv> [model_path] <input_csv>');
    console.error('  training_csv: Path to CSV file with training data');
    console.error('  model_path: (Optional) Path to save/load model');
    console.error('  input_csv: Path to CSV file with data to predict');
    process.exit(1);
}

// Parse command line arguments
const trainingPath = process.argv[2];
let modelPath = null;
let inputPath = null;

if (process.argv.length === 4) {
    // No model path specified
    inputPath = process.argv[3];
} else if (process.argv.length >= 5) {
    // Model path specified
    modelPath = process.argv[3];
    inputPath = process.argv[4];
}

// Verify file paths
if (!fs.existsSync(trainingPath)) {
    console.error(`Training file not found: ${trainingPath}`);
    process.exit(1);
}

if (!fs.existsSync(inputPath)) {
    console.error(`Input file not found: ${inputPath}`);
    process.exit(1);
}

// Random forest options
const options = {
    maxFeatures: 5,  // Using all features
    nEstimators: 100, // Number of trees in forest
    replacement: false,
    seed: 42
};

console.log('Starting housing price prediction...');
console.log(`Training data: ${trainingPath}`);
if (modelPath) console.log(`Model path: ${modelPath}`);
console.log(`Input data: ${inputPath}`);

// Create predictor instance
const predictor = new HousingPriceEstimator(trainingPath, modelPath, options);

// Run predictions
predictor.run(inputPath)
    .then(({ data, metrics, predictions, prices }) => {
        console.log('\nPrediction Results:');
        console.log('---------------------------------');
        
        // Display predictions
        data.forEach((features, i) => {
            console.log(`Property ${i+1}:`);
            console.log(`  Features: ${features.join(', ')}`);
            console.log(`  Predicted Price: $${Math.round(predictions[i]).toLocaleString()}`);
            console.log(`  Actual Price: $${Math.round(prices[i]).toLocaleString()}`);
            console.log(`  Difference: ${metrics.diffs[i]}`);
        });
        
        console.log('\nMetrics:');
        console.log('---------------------------------');
        console.log(`MAE: ${metrics.mae.toFixed(2)}`);
        console.log(`MSE: ${metrics.mse.toFixed(2)}`);
        console.log(`RMSE: ${metrics.rmse.toFixed(2)}`);
        console.log(`R-squared: ${metrics.r2.toFixed(2)}`);
        
        console.log('\nPrediction complete!');
    })
    .catch(err => {
        console.error('Error during prediction:', err);
        process.exit(1);
    });