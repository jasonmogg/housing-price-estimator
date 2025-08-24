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
        return Object.freeze(['bedrooms', 'bathrooms', 'livingArea', 'lotSize', 'state', 'lastSoldPrice']);
    }

    get labelField () {
        return 'streetAddress';
    }

    get predictionField () {
        return 'price';
    }
}

export default HousingPriceEstimator;
