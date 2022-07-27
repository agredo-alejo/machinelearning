import { forEachReturn, Tensor } from "@apjs/tensor";
import { Layer } from "./layer";


export interface DropoutLike {
    rate?: number;
    inputShape?: number | number[]
}
export class DropoutClass extends Layer {

    rate: number;
    inputShape: number | number[];
    outputShape: number | number[];

    constructor(object: DropoutLike = {}) {
        super()
        this.type = 'dropout'
        this.rate = object.rate || 0.1
        this.inputShape = object.inputShape ? object.inputShape : 1
        this.outputShape = this.inputShape
    }
    resize(inputShape: number | number[]) {
        this.inputShape = inputShape
        this.outputShape = this.inputShape
    }
    forward(input: Tensor[]) {
        this.input = input
        let result: Tensor[] = []
        for (let index = 0; index < input.length; index++) {
            result[index] = new Tensor(forEachReturn(input[index].data, x => Math.random() < this.rate ? 0 : x), input[index].shape)
        }
        return result
    }
    backward(outputGradient: Tensor[]) {
        return outputGradient
    }
    serialize() {
        return JSON.stringify(this)
    }
    save() {
        return {
            type: this.type,
            rate: this.rate,
            inputShape: this.inputShape
        }
    }
}