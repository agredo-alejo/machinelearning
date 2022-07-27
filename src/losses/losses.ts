import { Tensor } from "@apjs/tensor";
import { binaryCrossEntropy, binaryCrossEntropy_prime, crossEntropy, crossEntropy_prime, meanSquaredError, meanSquaredError_prime, softmaxCrossEntropy, softmaxCrossEntropy_prime } from "./operations";


export interface LossLike {
    forward: (output: Tensor, target: Tensor) => number;
    backward: (output: Tensor, target: Tensor) => Tensor;
}

export const meanSquaredErrorLoss: LossLike = {
    forward: meanSquaredError,
    backward: meanSquaredError_prime,
}

export const crossEntropyLoss: LossLike = {
    forward: crossEntropy,
    backward: crossEntropy_prime,
}

export const softmaxCrossEntropyLoss: LossLike = {
    forward: softmaxCrossEntropy,
    backward: softmaxCrossEntropy_prime
}

export const binaryCrossEntropyLoss: LossLike = {
    forward: binaryCrossEntropy,
    backward: binaryCrossEntropy_prime,
}

export const lossesHandler = (name: string) => {
    switch (name) {
        case 'mse':
            return meanSquaredErrorLoss
        case 'crossEntropy':
            return crossEntropyLoss
        case 'softmaxCrossEntropy':
            return softmaxCrossEntropyLoss
        case 'binaryCrossEntropy':
            return binaryCrossEntropyLoss
        default:
            return meanSquaredErrorLoss
    }
}