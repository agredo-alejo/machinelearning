import { fill, randomUniform, randomNormal } from "@apjs/tensor"
import { calculateFans } from "./utils";



export const constantInitializer = (shape: number[], value = 0) => {
    return fill(shape, value)
}


export const onesInitializer = (shape: number[]) => {
    return fill(shape, 1)
}


export const zerosInitializer = (shape: number[]) => {
    return fill(shape, 0)
}


export const randomUniformInitializer = (shape: number[], min?: number, max?: number) => {
    let limit = calculateFans(shape)[0] || 1
    limit = 1 / Math.sqrt(limit)
    let internMin = min || -limit
    let internMax = max || limit

    return randomUniform(shape, internMin, internMax)
}


export const randomNormalInitializer = (shape: number[], mean = 0, stddev = 1) => {

    return randomNormal(shape, mean, stddev)
}


export const xavierUniformInitializer = (shape: number[]) => {
    let ioFans = calculateFans(shape)
    let fanIn = ioFans[0] || 1
    let fanOut = ioFans[1] || 1
    let limit = Math.sqrt(6) / Math.sqrt(fanIn + fanOut)

    return randomUniform(shape, -limit, limit)
}


export const xavierNormalInitializer = (shape: number[]) => {
    let ioFans = calculateFans(shape)
    let fanIn = ioFans[0] || 1
    let fanOut = ioFans[1] || 1
    let stddev = Math.sqrt(2 / (fanIn + fanOut))

    return randomNormal(shape, 0, stddev)
}


export const heUniformInitializer = (shape: number[]) => {
    let ioFans = calculateFans(shape)
    let fanIn = ioFans[0] || 1
    let limit = Math.sqrt(6 / fanIn)

    return randomUniform(shape, -limit, limit)
}


export const heNormalInitializer = (shape: number[]) => {
    let ioFans = calculateFans(shape)
    let fanIn = ioFans[0] || 1
    let stddev = Math.sqrt(2 / fanIn)

    return randomNormal(shape, 0, stddev)
}


export const varianceScalingInitializer = (
    shape: number[],
    scale = 1,
    mode = 'fan_out',
    distribution = "uniform") => {
    let ioFans = calculateFans(shape)
    let fanIn = ioFans[0] || 1
    let fanOut = ioFans[1] || 1

    let n = fanIn
    if (mode == 'fan_out') {
        n = fanOut
    } else if (mode == 'avg') {
        n = (fanIn + fanOut) / 2
    }

    if (distribution == 'uniform') {
        let limit = Math.sqrt(3 * scale / n)
        return randomUniform(shape, -limit, limit)
    }

    let stddev = Math.sqrt(scale / n)
    return randomNormal(shape, 0, stddev)

}
export const initializersHanlder = (name?: string) => {
    switch (name) {
        case 'constant':
            return constantInitializer
        case 'ones':
            return onesInitializer
        case 'zeros':
            return zerosInitializer
        case 'randomUniform':
            return randomUniformInitializer
        case 'randomNormal':
            return randomNormalInitializer
        case "xavierUniform":
            return xavierUniformInitializer
        case "xavierNormal":
            return xavierNormalInitializer
        case "heUniform":
            return heUniformInitializer
        case "heNormal":
            return heNormalInitializer
        case "varianceScaling":
            return varianceScalingInitializer

        default:
            return randomUniformInitializer
    }
}