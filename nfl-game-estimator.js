import RandomForestEstimator from './random-forest-estimator.js';

export class HousingPriceEstimator extends RandomForestEstimator {
    get dataFilters () {
        return [
            { homeTeam: 'Washington Redskins' }
        ];
    }

    get featureFilters () {
        return Object.freeze({
            bathrooms: (value) => value >= 1
        });
    }

    get features () {
        return Object.freeze('awayTeam,awayavgScore,awayavgFirstDowns,awayavgTurnoversLost,awayavgPassingYards,awayavgRushingYards,awayavgOffensiveYards,awayavgPassingYardsAllowed,awayavgRushingYardsAllowed,awayavgTurnoversForced,awayavgYardsAllowed,awayavgOppScore,awayWins,awayStreak,awaySOS,homeavgScore,homeavgFirstDowns,homeavgTurnoversLost,homeavgPassingYards,homeavgRushingYards,homeavgOffensiveYards,homeavgPassingYardsAllowed,homeavgRushingYardsAllowed,homeavgTurnoversForced,homeavgYardsAllowed,homeavgOppScore,homeWins,homeStreak,homeSOS,Winner,actualSpread,week,season,awayRecordAgainstOpp,homeRecordAgainstOpp'.split(','));
    }

    get labelField () {
        return 'homeTeam';
    }

    get predictionField () {
        return 'Winner';
    }
}

export default HousingPriceEstimator;
