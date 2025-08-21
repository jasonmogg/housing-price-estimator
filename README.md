# Housing Price Estimator

A Node.js application that uses random forest algorithm to predict housing prices based on CSV data.

## Features

- Uses machine learning (Random Forest) to predict housing prices
- Supports training on CSV files containing housing data
- Can save and load trained models
- Provides price predictions for new properties

## Installation

```bash
npm install
```

## Usage

Run the program with the following command:

```bash
node index.js <training_csv> [model_path] <input_csv>
```

Where:
- `training_csv`: Path to CSV file with training data
- `model_path`: (Optional) Path to save/load model
- `input_csv`: Path to CSV file with data to predict

Example:
```bash
node index.js training2.csv model.json input2.csv
```

## CSV Format

The CSV file should contain at least the following columns:
- bedrooms
- bathrooms
- yearBuilt
- livingArea
- lotSize
- price (for training data)

## Requirements

- Node.js v14 or higher
- npm packages:
  - ml-random-forest
  - csvtojson
