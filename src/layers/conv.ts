import { Tensor, TensorLike4D, TensorObject, zeros, add, cc_paddingAmt, manyFiltersOutputShape, correlate2D, convolution2D, correlationManyFiltersAddBias } from "@apjs/tensor"
import { TrainableLayer } from "./layer";
import { initializersHanlder } from "../initializers";


export interface ConvLike {
    filters: number;
    padding?: string;
    weightsInitializer?: string;
    biasInitializer?: string;
    kernelSize: number | number[];
    inputShape?: number[];
    stride?: number | number[];
    dilation?: number | number[];
    weightsConstrain?: number[];
    biasConstrain?: number[];
    weights?: TensorObject;
    bias?: TensorObject;

}
export class ConvClass extends TrainableLayer {

    filters: number;
    padding: string;
    kernelSize: number[];
    inputShape: number[];
    stride: number[];
    dilation: number[];
    kernelShape: number[];
    padAmt: number[];
    kernelGradientPad: string | number[];
    inputGradientPad: string | number[];
    outputShape: number[];


    constructor(object: ConvLike) {
        super()
        this.type = 'conv'
        this.name = this.type
        this.isTrainable = true
        this.filters = Math.floor(object.filters)
        this.padding = object.padding || 'valid'
        if (object.weightsConstrain) this.weightsConstrain = object.weightsConstrain
        if (object.biasConstrain) this.biasConstrain = object.biasConstrain


        this.weightsInitializer = object.weightsInitializer || "xavierNormal"
        this.weightsInitializerFunction = initializersHanlder(this.weightsInitializer)


        this.biasInitializer = object.biasInitializer || "zeros"
        this.biasInitializerFunction = initializersHanlder(this.biasInitializer)


        this.kernelSize = typeof object.kernelSize == 'number' ? this.kernelSize = [object.kernelSize, object.kernelSize] : [object.kernelSize[0], object.kernelSize[1]]
        this.kernelSize.map(x => Math.floor(x))

        this.stride = [1, 1]
        if (object.stride) {
            this.stride = object.stride instanceof Array ? object.stride : [object.stride, object.stride]
            this.stride.map(x => Math.floor(x))
        }

        this.dilation = [1, 1]
        if (object.dilation) {
            this.dilation = object.dilation instanceof Array ? object.dilation : [object.dilation, object.dilation]
            this.dilation.map(x => Math.floor(x))
        }

        this.kernelGradientPad = 'valid'
        this.inputGradientPad = 'full'
        if (this.padding == 'same') {
            this.kernelGradientPad = [
                this.kernelSize[0] - 1,
                this.kernelSize[1] - 1
            ]
            this.inputGradientPad = 'same'
        }
        this.inputShape = object.inputShape ? object.inputShape.slice() : [1, 1, 1]
        this.inputShape.map(x => Math.floor(x))

        this.kernelShape = [this.filters, this.inputShape[0], this.kernelSize[0], this.kernelSize[1]]
        this.padAmt = cc_paddingAmt(this.inputShape, this.kernelShape, this.padding, this.stride, this.dilation)
        this.outputShape = manyFiltersOutputShape(this.inputShape, this.kernelShape, this.padAmt, this.stride, this.dilation)


        this.weights = this.weightsInitializerFunction(this.kernelShape)
        this.bias = this.biasInitializerFunction(this.outputShape)

        this.weightsGradient = zeros(this.weights.shape)
        this.biasGradient = zeros(this.bias.shape)
        this.optimizerWeights = [this.weightsGradient, zeros(this.weights.shape), zeros(this.weights.shape)]
        this.optimizerBias = [this.biasGradient, zeros(this.bias.shape), zeros(this.bias.shape)]
        this.parameters = this.weights.size() + this.bias.size()
        if (object.weights !== undefined) {
            this.setParameters(object)
        }

        // TODO: CALCULATE PADDING FOR BACKPROP WITH STRIDE AND DILATION !== 1 
    }
    resize(inputShape: number | number[]) {
        this.inputShape = inputShape instanceof Array ? inputShape : [1, 1, 1]
        this.inputShape.map(x => Math.floor(x))

        this.kernelShape = [this.filters, this.inputShape[0], this.kernelSize[0], this.kernelSize[1]]
        this.padAmt = cc_paddingAmt(this.inputShape, this.kernelShape, this.padding, this.stride, this.dilation)
        this.outputShape = manyFiltersOutputShape(this.inputShape, this.kernelShape, this.padAmt, this.stride, this.dilation)

        this.weights = this.weightsInitializerFunction(this.kernelShape)
        this.bias = this.biasInitializerFunction(this.outputShape)
        this.weightsGradient = zeros(this.weights.shape)
        this.biasGradient = zeros(this.bias.shape)
        this.optimizerWeights = [this.weightsGradient, zeros(this.weights.shape), zeros(this.weights.shape)]
        this.optimizerBias = [this.biasGradient, zeros(this.bias.shape), zeros(this.bias.shape)]
        this.parameters = this.weights.size() + this.bias.size()
    }
    forward(input: Tensor[]) {
        this.input = input

        for (let index = 0; index < this.input.length; index++) {
            this.output[index] = correlationManyFiltersAddBias(this.input[index], this.weights, this.bias, this.padAmt, this.stride, this.dilation)
        }
        return this.output
    }
    backward(outputGradient: Tensor[]) {
        let internOutputGradient = outputGradient

        let inputGradient: Tensor[] = []
        for (let index = 0; index < outputGradient.length; index++) {
            inputGradient[index] = zeros(this.inputShape)

            let kernelsGradient: Tensor | TensorLike4D = []

            for (let i = 0; i < this.kernelShape[0]; i++) {
                kernelsGradient[i] = []
                for (let j = 0; j < this.kernelShape[1]; j++) {

                    kernelsGradient[i][j] = correlate2D(this.input[index].data[j], internOutputGradient[index].data[i], this.kernelGradientPad).data
                    let currentConvolution = convolution2D(internOutputGradient[index].data[i], this.weights.data[i][j], this.inputGradientPad)
                    inputGradient[index].data[j] = add(inputGradient[index].data[j], currentConvolution).data
                }
            }

            kernelsGradient = new Tensor(kernelsGradient, this.kernelShape)
            this.weightsGradient.add(kernelsGradient)
            this.biasGradient.add(internOutputGradient[index])
        }
        return inputGradient
    }
    save() {
        return {
            type: this.type,
            filters: this.filters,
            padding: this.padding,
            weightsInitializer: this.weightsInitializer,
            biasInitializer: this.biasInitializer,
            kernelSize: this.kernelSize,
            inputShape: this.inputShape,
            stride: this.stride,
            dilation: this.dilation,
            weightsConstrain: this.weightsConstrain,
            biasConstrain: this.biasConstrain,
            weights: this.weights,
            bias: this.bias,
        }
    }
}