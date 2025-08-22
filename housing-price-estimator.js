import RandomForestEstimator from './random-forest-estimator.js';

export class HousingPriceEstimator extends RandomForestEstimator {
    get features () {
        return Object.freeze(['bedrooms', 'bathrooms', 'yearBuilt', 'livingArea', 'lotSize']);
    }

    get predictionField () {
        return 'price';
    }
}

export default HousingPriceEstimator;
