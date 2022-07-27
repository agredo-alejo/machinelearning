import { TrainableLayer } from "./layer";
import { dot, Tensor, TensorObject, transpose, zeros } from "@apjs/tensor";
import { initializersHanlder } from "../initializers";


export interface DenseLike {

    units: number;
    inputShape?: number;
    weightsInitializer?: string;
    biasInitializer?: string;
    weightsConstrain?: number[];
    biasConstrain?: number[];
    weights?: TensorObject;
    bias?: TensorObject;

}
export class DenseClass extends TrainableLayer {

    units: number;
    outputShape: number;
    inputShape: number;


    constructor(object: DenseLike) {
        super()
        this.type = 'dense'


        this.isTrainable = true
        this.units = Math.floor(object.units)
        this.outputShape = this.units


        this.weightsInitializer = object.weightsInitializer || "xavierNormal"
        this.weightsInitializerFunction = initializersHanlder(this.weightsInitializer)


        this.biasInitializer = object.biasInitializer || "zeros"
        this.biasInitializerFunction = initializersHanlder(this.biasInitializer)


        this.inputShape = object.inputShape ? Math.floor(object.inputShape) : 1
        this.weights = this.weightsInitializerFunction([this.units, this.inputShape])
        this.bias = this.biasInitializerFunction([this.units])
        this.weightsConstrain = object.weightsConstrain || [-1e7, 1e7]
        this.biasConstrain = object.biasConstrain || [-1e7, 1e7]

        this.weightsGradient = zeros(this.weights.shape)
        this.biasGradient = zeros(this.bias.shape)
        this.optimizerWeights = [this.weightsGradient, zeros(this.weights.shape), zeros(this.weights.shape)]
        this.optimizerBias = [this.biasGradient, zeros(this.bias.shape), zeros(this.bias.shape)]

        this.parameters = this.units * this.inputShape + this.outputShape
        if (object.weights !== undefined) {
            this.setParameters(object)
        }
    }
    resize(inputShape: number | number[]) {
        this.inputShape = inputShape instanceof Array ? inputShape[0] : inputShape
        this.weights = this.weightsInitializerFunction([this.units, this.inputShape])
        this.bias = this.biasInitializerFunction([this.units])

        this.reshapeOptimizerParameters()
        this.parameters = this.units * this.inputShape + this.outputShape
    }
    forward(input: Tensor[]) {
        this.input = input
        for (let index = 0; index < this.input.length; index++) {
            this.output[index] = dot(this.weights, this.input[index]).add(this.bias)
        }
        return this.output
    }
    backward(outputGradient: Tensor[]) {
        let transposedWeights = transpose(this.weights)
        let inputGradient: Tensor[] = []

        for (let index = 0; index < outputGradient.length; index++) {
            let internOutputGradient = outputGradient[index]


            let transposedInputs = transpose(this.input[index])

            let weightsDeltas = dot(internOutputGradient, transposedInputs)

            this.weightsGradient.add(weightsDeltas)
            this.biasGradient.add(internOutputGradient)


            inputGradient[index] = dot(transposedWeights, internOutputGradient)
        }
        return inputGradient
    }
    save() {
        return {
            type: this.type,
            weights: this.weights,
            bias: this.bias,
            weightsInitializer: this.weightsInitializer,
            biasInitializer: this.biasInitializer,
            inputShape: this.inputShape,
            units: this.units,
            outputShape: this.outputShape,
            weightsConstrain: this.weightsConstrain,
            biasConstrain: this.biasConstrain,
        }
    }
}
