import { add, div, divNoNan, exp, log, max, mean, mult, size, sqd, sub, sum, sumExp, Tensor, TensorLike } from "@apjs/tensor"


export const meanSquaredError = (output: Tensor | TensorLike, target: Tensor | TensorLike) => {
    return mean(sqd(target, output))
}
export const meanSquaredError_prime = (output: Tensor | TensorLike, target: Tensor | TensorLike) => {
    return div(
        mult(2, sub(output, target)),
        size(output)
    )
}

export const crossEntropy = (output: Tensor | TensorLike, target: Tensor | TensorLike) => {
    return -sum(mult(target, log(output)))
}
export const crossEntropy_prime = (output: Tensor | TensorLike, target: Tensor | TensorLike) => {
    return mult(target, divNoNan(-1, output))
}


export const softmaxCrossEntropy = (output: Tensor | TensorLike, target: Tensor | TensorLike) => {
    
    let trick = sub(output, max(output))
    let activation = div(exp(trick), sumExp(trick))

    return -sum(mult(target, log(activation)))
}

export const softmaxCrossEntropy_prime = (output: Tensor | TensorLike, target: Tensor | TensorLike) => {
    let trick = sub(output, max(output))

    let activation = divNoNan(exp(trick), sumExp(trick))
    let result = sub(activation, target)

    return result
}

export const binaryCrossEntropy = (output: Tensor | TensorLike, target: Tensor | TensorLike) => {
    let first = mult(target, log(output))
    let second = sub(1, target)
    let third = log(sub(1, output))
    let result = add(first, mult(second, third))
    return -mean(result)
}
export const binaryCrossEntropy_prime = (output: Tensor | TensorLike, target: Tensor | TensorLike) => {
    let first = divNoNan(sub(1, target), sub(1, output))
    let second = divNoNan(target, output)
    let result = sub(first, second)
    return divNoNan(result, size(result))
}