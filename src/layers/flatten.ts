import { Tensor, TensorLike3D } from "@apjs/tensor"
import { Layer } from "./layer";


export interface FlattenLike {
    inputShape?: number[];
    outputShape?: number;
}

export class FlattenCLass extends Layer {

    inputShape: number[];
    outputShape: number;

    constructor(object: FlattenLike = {}) {
        super()
        this.type = 'flatten'
        this.inputShape = object.inputShape instanceof Array ? object.inputShape.slice() : [1, 1, 1]
        this.inputShape.map(x => Math.floor(x))
        this.outputShape = 1
        for (let i = 0; i < this.inputShape.length; i++) {
            this.outputShape *= this.inputShape[i]
        }
    }
    resize(inputShape: number | number[]) {
        this.inputShape = inputShape instanceof Array ? inputShape.slice() : [1, 1, 1]
        this.inputShape.map(x => Math.floor(x))
        this.outputShape = 1
        for (let i = 0; i < this.inputShape.length; i++) {
            this.outputShape *= this.inputShape[i]
        }
    }
    forward(input: Tensor[]) {
        this.input = input
        let result: Tensor[] = []
        for (let index = 0; index < input.length; index++) {
            let vector: number[] = []
            input[index].forEach(x => {
                vector.push(x)
                return x
            })
            result[index] = new Tensor(vector, [this.outputShape])
        }
        return result
    }
    backward(outputGradient: Tensor[]) {
        let inputGradient: Tensor[] = []
        for (let index = 0; index < outputGradient.length; index++) {

            let result: TensorLike3D = []
            let element = 0
            for (let depth = 0; depth < this.inputShape[0]; depth++) {
                result[depth] = []
                for (let row = 0; row < this.inputShape[1]; row++) {
                    result[depth][row] = []
                    for (let col = 0; col < this.inputShape[2]; col++) {
                        result[depth][row][col] = outputGradient[index].data[element]
                        element++
                    }
                }
            }
            inputGradient[index] = new Tensor(result, this.inputShape)
        }
        return inputGradient
    }
    serialize() {
        return JSON.stringify(this)
    }
    save(){
        return {
            type: this.type,
            inputShape: this.inputShape
        }
    }
}
