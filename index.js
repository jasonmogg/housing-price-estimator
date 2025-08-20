import HousingPriceEstimator from './housing-price-estimator.js';

new HousingPriceEstimator(process.argv[2], process.argv[3]).run(process.argv[4]).then(({ data, predictions }) => {
    data.forEach((row, index) => console.log(row, predictions[index]));
});
