import { copy, fromObject, shapesEqual, Tensor, TensorLike, zeros, _randomRangeArray } from "@apjs/tensor";
import { initializersHanlder } from "../initializers";
import { Optimizer } from "../optimizers";

export class Layer {

    input: Tensor[];
    type: string;
    isTrainable: boolean;
    name: string;
    output: Tensor[];

    constructor() {
        this.input = [new Tensor([])]
        this.output = [new Tensor([])]
        this.type = ''
        this.name = ''
        this.isTrainable = false
    }
    forward(input: Tensor[]): Tensor[] {
        return input
    }
    backward(outputGradient: Tensor[]): Tensor[] {
        return outputGradient
    }

    serialize() {
        return JSON.stringify(this)
    }
}
export class TrainableLayer extends Layer {

    weights: Tensor;
    bias: Tensor;
    weightsGradient: Tensor;
    biasGradient: Tensor;
    biasConstrain: number[];
    weightsConstrain: number[];
    optimizerWeights: Tensor[];
    optimizerBias: Tensor[];
    weightsInitializer: string;
    weightsInitializerFunction: Function;
    biasInitializer: string;
    biasInitializerFunction: Function;
    parameters: number;


    constructor() {
        super()
        this.name = this.type
        this.weights = new Tensor([])
        this.bias = new Tensor([])
        this.weightsConstrain = [-1e7, 1e7]
        this.biasConstrain = [-1e7, 1e7]
        this.weightsGradient = new Tensor([])
        this.biasGradient = new Tensor([])
        this.optimizerWeights = [this.weightsGradient]
        this.optimizerBias = [this.weightsGradient]
        this.weightsInitializer = "xavierNormal"
        this.weightsInitializerFunction = initializersHanlder(this.weightsInitializer)
        this.biasInitializer = "zeros"
        this.biasInitializerFunction = initializersHanlder(this.biasInitializer)
        this.parameters = 0
    }
    reshapeOptimizerParameters() {
        this.weightsGradient = zeros(this.weights.shape)
        this.biasGradient = zeros(this.bias.shape)
        this.optimizerWeights = [this.weightsGradient, zeros(this.weights.shape), zeros(this.weights.shape)]
        this.optimizerBias = [this.biasGradient, zeros(this.bias.shape), zeros(this.bias.shape)]
    }
    update(optimizer: Optimizer, batchSize = 1, index = 0) {
        this.weightsGradient.divNoNan(batchSize)
        this.biasGradient.divNoNan(batchSize)

        let weightsGradient = optimizer.apply(this.optimizerWeights, index)
        let biasGradient = optimizer.apply(this.optimizerBias, index)


        this.weights.sub(weightsGradient)
        this.bias.sub(biasGradient)

        this.weights.constrain(this.weightsConstrain)
        this.bias.constrain(this.biasConstrain)

        this.weightsGradient.mult(0)
        this.biasGradient.mult(0)
    }
    setParameters(object: any) {
        this.setWeights(fromObject(object.weights))
        this.setBias(fromObject(object.bias))
        // this.weightsInitializerFunction = initializersHanlder(object.kernelInitializer)
        // this.biasInitializerFunction = initializersHanlder(object.biasInitializer)
        // for (let index = 1; index < object.optimizerWeights.length; index++) {
        //     this.optimizerWeights[index] = copy(fromObject(object.optimizerWeights[index]))
        //     this.optimizerBias[index] = copy(fromObject(object.optimizerBias[index]))
        // }
    }
    setWeights(weights: Tensor | TensorLike) {
        let newWeights = copy(weights)
        if (!shapesEqual(newWeights.shape, this.weights.shape)) throw "Weights shapes don't match";
        this.weights = newWeights
    }
    getWeights() {
        return this.weights.copy()
    }
    setBias(bias: Tensor | TensorLike) {
        let newBias = copy(bias)
        if (!shapesEqual(newBias.shape, this.bias.shape)) throw "Bias shapes don't match";
        this.bias = newBias
    }
    getBias() {
        return this.bias.copy()
    }
    mutate(rate: number = 0.1, range: number[]) {
        this.mutateWeights(rate, range)
        this.mutateBias(rate, range)
    }
    mutateWeights(rate: number, range: number[]) {
        this.weights.map(x => Math.random() < rate ? _randomRangeArray(range) : x)
    }
    mutateBias(rate: number, range: number[]) {
        this.bias.map(x => Math.random() < rate ? _randomRangeArray(range) : x)
    }

    save(){
        return {
            weights: this.getWeights(),
            bias: this.getBias()
        }
    }
}
