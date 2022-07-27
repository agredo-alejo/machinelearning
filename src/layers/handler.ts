import { DenseClass, DenseLike } from "./dense"
import { ConvClass, ConvLike } from "./conv"
import { FlattenCLass, FlattenLike } from "./flatten"
import { DropoutClass, DropoutLike } from "./dropout"
import { NormClass, NormLike } from "./norm"
import { ActivationClass } from "./activation"
import { MaxPoolingClass, MpLike } from "./maxPooling"
import { ActivationLike, activationsHandler } from "../activations"



export const layersHandler = (type: string | undefined) => {

    switch (type) {
        case 'dense':
            return DenseClass
        case 'conv':
            return ConvClass
        case 'maxpooling':
            return MaxPoolingClass
        case 'flatten':
            return FlattenCLass
        case 'dropout':
            return DropoutClass
        case 'norm':
            return NormClass
        case 'activation':
            return ActivationClass
        default:
            return DenseClass
    }
}

export const layerFromObject = (object: any) => {
    let layerConstructor = layersHandler(object.type)
    return new layerConstructor(object)
}

export const denseLayer = (object: DenseLike) => {
    return new DenseClass(object)
}
export const convLayer = (object: ConvLike) => {
    return new ConvClass(object)
}
export const maxPoolingLayer = (object: MpLike) => {
    return new MaxPoolingClass(object)
}
export const normLayer = (object: NormLike) => {
    return new NormClass(object)
}
export const flattenLayer = (object: FlattenLike) => {
    return new FlattenCLass(object)
}
export const dropoutLayer = (object: DropoutLike) => {
    return new DropoutClass(object)
}
export const activationLayer = (name: string) => {
    const activation = activationsHandler(name)
    return new ActivationClass(activation)
}


export type LayerLike = DenseLike | ConvLike | MpLike | FlattenLike | DropoutLike | ActivationLike | NormLike

export type LayerClasses = DenseClass | ConvClass | MaxPoolingClass | FlattenCLass | DropoutClass | ActivationClass | NormClass

export type TrainableLayers = DenseClass | ConvClass | NormClass