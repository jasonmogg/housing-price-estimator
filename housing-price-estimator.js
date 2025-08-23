import RandomForestEstimator from './random-forest-estimator.js';

export class HousingPriceEstimator extends RandomForestEstimator {
    actualFilter (actual) {
        return actual > 0;
    }

    get featureFilters () {
        return Object.freeze({
            bathrooms: (value) => value >= 1
        });
    }

    get features () {
        return Object.freeze(['bathrooms', 'yearBuilt', 'livingArea', 'state']);
    }

    get predictionField () {
        return 'price';
    }
}

export default HousingPriceEstimator;
