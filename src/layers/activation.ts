import { Tensor } from "@apjs/tensor";
import { ActivationLike } from "../activations";
import { Layer } from "./layer";


export class ActivationClass extends Layer {

    forwardedInput: Tensor[];
    inputShape: number | number[];
    outputShape: number | number[];
    activation: ActivationLike

    constructor(object: ActivationLike) {
        super()
        this.isTrainable = false
        this.type = 'activation'
        this.name = object.name
        this.activation = object

        this.inputShape = 0
        this.outputShape = this.inputShape
        this.forwardedInput = [new Tensor([])]
    }
    resize(inputShape: number | number[]) {
        this.inputShape = inputShape
        this.outputShape = this.inputShape
    }
    forward(input: Tensor[]) {
        this.input = input
        for (let index = 0; index < input.length; index++) {
            this.forwardedInput[index] = this.activation.forward(input[index])
        }
        return this.forwardedInput
    }
    backward(outputGradient: Tensor[]) {
        let inputGradient: Tensor[] = []
        for (let index = 0; index < outputGradient.length; index++) {
            inputGradient[index] = this.activation.updateError(
                outputGradient[index], this.activation.backward(this.forwardedInput[index])
            )
        }
        return inputGradient
    }
    serialize() {
        return JSON.stringify(this)
    }
    save(){
        return {
            type: this.type,
            name: this.name,
            inputShape: this.inputShape
        }
    }
}

