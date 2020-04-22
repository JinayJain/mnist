import * as tf from "@tensorflow/tfjs";

let model;
async function loadModel() {
    model = await tf.loadLayersModel("model/model.json");
    console.log("MNIST model loaded.");
}
loadModel();

function getPixel(pixels, size, i, j) {
    return pixels[4 * (size * i + j) + 3];
}

// naive image downsizing algorithm
function preprocess(pixels, size, newSize) {
    const kSize = Math.round(size / newSize);

    let retPixels = Array(newSize * newSize).fill(0);

    for (let i = 0; i < newSize; i++) {
        for (let j = 0; j < newSize; j++) {
            let nearest = {
                j: Math.round(j * (size / newSize)),
                i: Math.round(i * (size / newSize)),
            };

            let halfKSize = Math.round(kSize / 2);
            let kernel = {
                top: Math.max(0, nearest.i - halfKSize),
                left: Math.max(0, nearest.j - halfKSize),
                bottom: Math.min(size - 1, nearest.i + halfKSize),
                right: Math.min(size - 1, nearest.j + halfKSize),
            };

            let numK = 0;

            let avg = 0;
            for (let k = kernel.top; k <= kernel.bottom; k++) {
                for (let l = kernel.left; l <= kernel.right; l++) {
                    avg += getPixel(pixels, size, k, l);
                    numK++;
                }
            }
            avg /= numK;
            retPixels[i * newSize + j] = avg;
        }
    }

    return tf.tensor4d(retPixels, [1, 28, 28, 1]);
}

export default async function recognizeDigit(pixels, size) {
    if (model == null) return "loading";
    const image = preprocess(pixels, size, 28);

    const preds = await model.predict(image).as1D().array();
    let prediction = 0;
    for (let i = 0; i < 10; i++) {
        if (preds[i] > preds[prediction]) {
            prediction = i;
        }
    }

    return prediction;
}
