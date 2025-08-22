import RandomForestEstimator from './random-forest-estimator.js';

export class HousingPriceEstimator extends RandomForestEstimator {
    get categoricalValues () {
        return Object.freeze({
            state: [ 'AL', 'AK', 'AS', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FM', 'FL', 'GA', 'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MH', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'MP', 'OH', 'OK', 'OR', 'PW', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VI', 'VA', 'WA', 'WV', 'WI', 'WY' ]
        });
    }

    actualFilter (actual) {
        return actual > 0;
    }

    get featureFilters () {
        return Object.freeze({
            bedrooms: (value) => value >= 1,
            bathrooms: (value) => value >= 1
        });
    }

    get features () {
        return Object.freeze(['bedrooms', 'bathrooms', 'yearBuilt', 'livingArea', 'lotSize', 'state']);
    }

    get predictionField () {
        return 'price';
    }
}

export default HousingPriceEstimator;
