import { Tensor, TensorObject, mult, sub, varianceSquared, add, divNoNan } from "@apjs/tensor";
import { initializersHanlder } from "../initializers";
import { TrainableLayer } from "./layer";


export interface NormLike {
    inputShape?: number | number[];
    weights?: TensorObject;
    bias?: TensorObject;
    weightsInitializer?: string;
    biasInitializer?: string;
}
export class NormClass extends TrainableLayer {

    inputShape: number | number[];
    internInputShape: number[];
    norm: Tensor;
    xmu: Tensor;
    variance: number;
    outputShape: number | number[];


    constructor(object: NormLike = {}) {
        super()

        this.type = 'norm'
        this.isTrainable = true
        this.inputShape = 1
        if (object.inputShape instanceof Array) this.inputShape = object.inputShape.slice().map(x => Math.floor(x))
        if (typeof object.inputShape == 'number') this.inputShape = Math.floor(object.inputShape)
        this.outputShape = this.inputShape
        this.internInputShape = this.inputShape instanceof Array ? this.inputShape.slice().map(x => Math.floor(x)) : [Math.floor(this.inputShape)]
        

        this.weightsInitializer = object.weightsInitializer || "xavierNormal"
        this.weightsInitializerFunction = initializersHanlder(this.weightsInitializer)


        this.biasInitializer = object.biasInitializer || "zeros"
        this.biasInitializerFunction = initializersHanlder(this.biasInitializer)

        // weights = gamma
        this.weights = this.weightsInitializerFunction(this.internInputShape)
        // bias = beta
        this.bias = this.biasInitializerFunction(this.internInputShape)
        this.reshapeOptimizerParameters()
        this.norm = new Tensor([])
        this.xmu = new Tensor([])
        this.variance = 0

        if (object.weights !== undefined) {
            this.setParameters(object)
        }
    }
    resize(inputShape: number | number[]) {
        this.inputShape = inputShape instanceof Array ? inputShape.slice().map(x => Math.floor(x)) : Math.floor(inputShape)
        this.outputShape = this.inputShape
        this.internInputShape = inputShape instanceof Array ? inputShape.slice().map(x => Math.floor(x)) : [Math.floor(inputShape)]
        // weights = gamma
        this.weights = this.weightsInitializerFunction(this.internInputShape)
        // bias = beta
        this.bias = this.biasInitializerFunction(this.internInputShape)
        this.reshapeOptimizerParameters()
        this.parameters = this.weights.size() + this.bias.size()
    }
    forward(input: Tensor[]) {
        this.input = input
        for (let index = 0; index < input.length; index++) {
            this.xmu = sub(this.input[index], this.input[index].mean())
            this.variance = varianceSquared(this.input[index]) + 1e-7
            this.norm = divNoNan(this.xmu, this.variance ** 0.5)
            let output = mult(this.weights, this.norm).add(this.bias)
            this.output[index] = output
        }
        return this.output
    }
    backward(outputGradient: Tensor[]) {
        let inputGradient: Tensor[] = []
        for (let index = 0; index < outputGradient.length; index++) {

            let internOutputGradient = this.output[index]

            let gammaGradient = mult(internOutputGradient, this.norm)

            this.weightsGradient.add(gammaGradient)
            this.biasGradient.add(internOutputGradient)

            // Inputs gradient
            let var_inv = 1 / this.variance ** 0.5
            inputGradient[index] = mult(internOutputGradient, this.weights)
            let varianceGradient = mult(inputGradient[index], this.xmu).mult(-0.5).mult(this.variance ** (-3 / 2))

            let muGradient = mult(inputGradient[index], -var_inv).add(mult(varianceGradient, divNoNan(1 / outputGradient.length, mult(this.xmu, -2))))

            inputGradient[index] = mult(inputGradient[index], var_inv).add(add(divNoNan(muGradient, outputGradient.length), divNoNan(mult(varianceGradient, 2 / outputGradient.length), this.xmu)))

        }
        return inputGradient
    }
    save() {
        return {
            type: this.type,
            inputShape: this.inputShape,
            weights: this.weights,
            bias: this.bias,
            weightsInitializer: this.weightsInitializer,
            biasInitializer: this.biasInitializer,
        }
    }
}