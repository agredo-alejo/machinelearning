import { add, divNoNan, mult, square, sub, Tensor } from "@apjs/tensor";

export type OptimizerFunction = (varUpdate: Tensor, gradientParams: Tensor[], index?: number) => void
export type CompleteFunction = { fn: (tensor: Tensor) => Tensor, fnPrime: (tensor: Tensor) => Tensor, computeGradient: (tensor: Tensor, fnDeriv: Tensor) => Tensor }

export class Optimizer {
    args: number[]; name: string
    constructor() {
        this.args = []
        this.name = ""
    }
    apply(gradientParams: Tensor[], index = 0) {
        return mult(gradientParams[0], index)
    }
}
export class Sgd extends Optimizer {
    constructor(args: number[]) {
        super()
        this.name = "sgd"
        this.args = args
    }
    apply(gradientParams: Tensor[]) {
        return mult(gradientParams[0], this.args[0])
    }
}
export const sgd = (learningRate = 0.1) => new Sgd([learningRate])

export class Momentum extends Optimizer {
    constructor(args: number[]) {
        super()
        this.name = "momentum"
        this.args = args
    }
    apply(gradientParams: Tensor[]) {
        // 0 = g,  1 = vd, 
        if (gradientParams[1] == undefined) {
            return mult(mult(gradientParams[0], 1 - this.args[1]), this.args[0])
        }
        gradientParams[1] = mult(gradientParams[1], this.args[1]).add(mult(gradientParams[0], 1 - this.args[1]))
        return mult(gradientParams[1], this.args[0])
    }
}
export const momentum = (learningRate = 0.1, momentum = 0.9) => new Momentum([learningRate, momentum])

export class Rmsprop extends Optimizer {
    constructor(args: number[]) {
        super()
        this.name = "rmsprop"
        this.args = args
    }
    apply(gradientParams: Tensor[]) {
        // 0 = g,  1 = sd, 
        if (gradientParams[1] == undefined) {
            let sd = mult(square(gradientParams[0]), 1 - this.args[1])
            return mult(divNoNan(gradientParams[0], add(sd, this.args[2]).sqrt()), this.args[0])
        } else {
            gradientParams[1] = mult(gradientParams[1], this.args[1]).add(mult(square(gradientParams[0]), 1 - this.args[1]))
            return mult(divNoNan(gradientParams[0], add(gradientParams[1], this.args[2]).sqrt()), this.args[0])
        }
    }
}
export const rmsprop = (learningRate = 0.01, momentum = 0.9, epsilon = 1e-7) => new Rmsprop([learningRate, momentum, epsilon])

export class Adam extends Optimizer {
    constructor(args: number[]) {
        super()
        this.name = "adam"
        this.args = args
    }
    apply(gradientParams: Tensor[], index = 0) {
        // 0 = g, 1 = vd,  2 = sd, 
        if (gradientParams[1] == undefined) {
            let vd = mult(gradientParams[0], 1 - this.args[1])
            let sd = mult(square(gradientParams[0]), 1 - this.args[2])

            let vdcorrected = divNoNan(vd, 1 - this.args[1])
            let sdcorrected = divNoNan(sd, 1 - this.args[2])

            return mult(divNoNan(vdcorrected, add(sdcorrected, this.args[3]).sqrt()), this.args[0])
        } else {
            gradientParams[1] = add(mult(gradientParams[1], this.args[1]), mult(gradientParams[0], 1 - this.args[1]))
            gradientParams[2] = add(mult(gradientParams[2], this.args[2]), mult(square(gradientParams[0]), 1 - this.args[2]))

            let vdcorrected = divNoNan(gradientParams[1], 1 - this.args[1] ** index)
            let sdcorrected = divNoNan(gradientParams[2], 1 - this.args[2] ** index)

            return mult(divNoNan(vdcorrected, add(sdcorrected, this.args[3]).sqrt()), this.args[0])
            // varUpdate.set(updated)
        }

    }
}
export const adam = (learningRate = 0.1, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7) => new Adam([learningRate, beta1, beta2, epsilon])

export const optimizersHandler = (object: any) => {
    switch (object.name) {
        case 'sgd':
            return sgd(...object.args)
        case 'momentum':
            return momentum(...object.args)
        case 'rmsprop':
            return rmsprop(...object.args)
        case 'adam':
            return adam(...object.args)
        default:
            return sgd()
    }
}
