import { Tensor, ones, sigmoid, sigmoid_prime, relu, relu_prime, leakyRelu, leakyRelu_prime, elu, elu_prime, softplus, softplus_prime, binaryStep, binaryStep_prime, tanh, tanh_prime, softmax, softmax_prime, mult, dot, } from "@apjs/tensor"


export type updateError = (outputGradient: Tensor, fnDeriv: Tensor) => Tensor

export const multUpdateError: updateError = (outputGradient, fnDeriv) => mult(outputGradient, fnDeriv)

export const dotUpdateError: updateError = (outputGradient, fnDeriv) => dot(fnDeriv, outputGradient)

export interface ActivationLike {
    forward: (t: Tensor) => Tensor; backward: (t: Tensor) => Tensor; updateError: updateError; name: string
}
export const linearActivation: ActivationLike = {
    forward: tensor => tensor,
    backward: tensor => ones(tensor.shape),
    updateError: (outputGradient: Tensor) => outputGradient,
    name: "linear"
}
export const sigmoidActivation: ActivationLike = {
    forward: tensor => sigmoid(tensor),
    backward: tensor => sigmoid_prime(tensor),
    updateError: multUpdateError,
    name: "sigmoid"
}
export const reluActivation: ActivationLike = {
    forward: tensor => relu(tensor),
    backward: tensor => relu_prime(tensor),
    updateError: multUpdateError,
    name: "relu"
}
export const leakyreluActivation: ActivationLike = {
    forward: tensor => leakyRelu(tensor),
    backward: tensor => leakyRelu_prime(tensor),
    updateError: multUpdateError,
    name: "leakyrelu"
}
export const eluActivation: ActivationLike = {
    forward: tensor => elu(tensor),
    backward: tensor => elu_prime(tensor),
    updateError: multUpdateError,
    name: "elu"
}
export const softplusActivation: ActivationLike = {
    forward: tensor => softplus(tensor),
    backward: tensor => softplus_prime(tensor),
    updateError: multUpdateError,
    name: "softplus"
}
export const binarystepActivation: ActivationLike = {
    forward: tensor => binaryStep(tensor),
    backward: tensor => binaryStep_prime(tensor),
    updateError: multUpdateError,
    name: "binarystep"
}
export const tanhActivation: ActivationLike = {
    forward: tensor => tanh(tensor),
    backward: tensor => tanh_prime(tensor),
    updateError: multUpdateError,
    name: "tanh"
}
export const softmaxActivation: ActivationLike = {
    forward: tensor => softmax(tensor),
    backward: tensor => softmax_prime(tensor),
    updateError: dotUpdateError,
    name: "softmax"
}

export const activationsHandler = (name: string) => {
    switch (name) {
        case "linear":
            return linearActivation
        case "sigmoid":
            return sigmoidActivation
        case "relu":
            return reluActivation
        case "leakyrelu":
            return leakyreluActivation
        case "elu":
            return eluActivation
        case "softplus":
            return softplusActivation
        case "binarystep":
            return binarystepActivation
        case "tanh":
            return tanhActivation
        case "softmax":
            return softmaxActivation

        default:
            return sigmoidActivation
    }
}
