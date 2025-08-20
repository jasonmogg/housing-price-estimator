import fs from 'node:fs';
import readline from 'node:readline';

class HousingPriceEstimator {
    #data;
    #labels;
    #loadPromise;

    constructor (labelsPath, dataPath) {
        this.#loadPromise = Promise.all([
            this.#readFirstLine(labelsPath).then(line => this.#labels = line.split(',')),
            import(dataPath, { with: { type: 'json' } }).then(module => this.#data = module.default)
        ]);
    }

    async #readFirstLine (filePath) {
        const fileStream = fs.createReadStream(filePath);
        const rl = readline.createInterface({
            input: fileStream,
            crlfDelay: Infinity
        });

        let result;

        // Get first line and close stream
        for await (const line of rl) {
            result = line;

            rl.close();

            break;
        }

        return result;
    }

    run () {
        this.#loadPromise.then(() => {
            console.log(this.#labels);
        });
    }
}

new HousingPriceEstimator(process.argv[2], process.argv[3]).run();
