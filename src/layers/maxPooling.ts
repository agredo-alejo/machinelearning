import { Tensor, TensorLike3D, zeros, maxPooling3D, mpPaddingAmt, mpOutputShape3D } from "@apjs/tensor"
import { Layer } from "./layer"


export interface MpLike {
    poolSize?: number | number[];
    stride?: number | number[];
    padding?: boolean;
    inputShape?: number[];
    coords?: number[][][];
}
export class MaxPoolingClass extends Layer {
    poolSize: number[];
    stride: number[];
    padding: boolean;
    padAmt: number[];
    inputShape: number[];
    coords: TensorLike3D;
    outputShape: number[]
    constructor(object: MpLike = {}) {
        super()
        this.type = 'maxpooling'

        this.poolSize = [2, 2]
        if (object.poolSize) {
            this.poolSize = object.poolSize instanceof Array ? object.poolSize : [object.poolSize, object.poolSize]
            this.poolSize.map(x => Math.floor(x))
        }

        this.stride = this.poolSize
        if (object.stride) {
            this.stride = object.stride instanceof Array ? object.stride : [object.stride, object.stride]
            this.stride.map(x => Math.floor(x))
        }

        this.padding = object.padding || false
        this.inputShape = object.inputShape ? object.inputShape.slice() : [1, 1, 1]
        this.inputShape.map(x => Math.floor(x))
        this.padAmt = mpPaddingAmt(this.inputShape, this.padding, this.poolSize, this.stride)
        this.outputShape = mpOutputShape3D(this.inputShape, this.padAmt, this.poolSize, this.stride)
        this.coords = object.coords ? object.coords.slice() : []

    }
    resize(inputShape: number | number[]) {
        this.inputShape = inputShape instanceof Array ? inputShape.slice() : [1, 1, 1]
        this.inputShape.map(x => Math.floor(x))
        this.padAmt = mpPaddingAmt(this.inputShape, this.padding, this.poolSize, this.stride)
        this.outputShape = mpOutputShape3D(this.inputShape, this.padAmt, this.poolSize, this.stride)
    }
    forward(input: Tensor[]) {
        this.input = input
        let result: Tensor[] = []
        for (let index = 0; index < input.length; index++) {
            this.coords[index] = []
            result[index] = maxPooling3D(this.input[index], this.padding, this.poolSize, this.stride, this.coords[index])
        }
        return result
    }
    backward(outputGradient: Tensor[]) {
        let result: Tensor[] = []
        for (let index = 0; index < this.input.length; index++) {
            result[index] = zeros(this.inputShape)
            for (let i = 0; i < this.coords[index].length; i++) {
                result[index].data[this.coords[index][i][0]][this.coords[index][i][1]][this.coords[index][i][2]] =
                    outputGradient[index].data[this.coords[index][i][0]][this.coords[index][i][3]][this.coords[index][i][4]]
            }
        }
        return result
    }
    save(){
        return {
            type: this.type,
            poolSize: this.poolSize,
            stride: this.stride,
            padding: this.padding,
            inputShape: this.inputShape,
            coords: this.coords,
        }
    }
}
