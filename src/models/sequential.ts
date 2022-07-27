import { shuffleMatch, Tensor } from "@apjs/tensor";
import { lossesHandler, LossLike } from "../losses";
import { Optimizer, optimizersHandler, sgd } from "../optimizers";
import { activationLayer, LayerClasses, layerFromObject, TrainableLayer, TrainableLayers } from "../layers";


interface CompileLike {
    loss: string;
    optimizer: Optimizer;
    trainDataSize?: number;
    batchSize?: number
}

export class Sequential {

    layers: LayerClasses[];
    numLayers: number;
    trainableLayers: TrainableLayers[];
    error: number;
    loss: string;
    lossFunction: LossLike;
    optimizer: Optimizer;
    parameters: number;
    batch: number;
    batchSize: number

    constructor() {
        this.batch = 0
        this.parameters = 0
        this.layers = []
        this.numLayers = 0
        this.loss = 'mse'
        this.lossFunction = lossesHandler(this.loss)
        this.trainableLayers = []
        this.error = 1
        this.optimizer = sgd(0.1)
        this.batchSize = 0
    }
    setLossFunction(lossName: string) {
        this.loss = lossName
        this.lossFunction = lossesHandler(this.loss)
    }
    compile(object: CompileLike) {
        this.optimizer = object.optimizer || sgd(0.1)

        this.loss = object.loss || "mse"
        this.lossFunction = lossesHandler(this.loss)
    }
    add(object: LayerClasses, fromSerialized: boolean = false) {
        if (this.numLayers > 0 && fromSerialized == false) {
            let inputShape = this.layers[this.numLayers - 1].outputShape
            object.resize(inputShape)
        }
        if (object instanceof TrainableLayer) {
            this.trainableLayers.push(object)
            this.parameters += object.parameters
        }
        this.layers.push(object)
        this.numLayers++
    }
    
   async predictOnBatch(input: Tensor[]){
        let output = input 

        for (let i = 0; i < this.numLayers; i++) {
            output = this.layers[i].forward(output)
        }
        
        return output
    }

    async predict(input: Tensor) {
        let result = await this.predictOnBatch([input])
        return result[0]
    }

    async train(inputs: Tensor[], targets: Tensor[], args = { epochs: 1, batchSize: 1, shuffle: false}) {
        let shuffle = args.shuffle || false
        let epochs = args.epochs || 1
        this.batchSize = args.batchSize || inputs.length

        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let batch = 0; batch < inputs.length / this.batchSize; batch++) {
                let internInputs = inputs.slice(batch, batch + this.batchSize)
                let internTargets = targets.slice(batch, batch + this.batchSize)

                await this.trainOnBatch(internInputs, internTargets, shuffle)
            }
        }
        if (this.batch > inputs.length / this.batchSize) {
            this.batch = 0
        }
    }
    async trainOnBatch(inputs: Tensor[], targets: Tensor[], shuffle = false) {
        let batchSize = inputs.length

        if (this.batchSize == 0) {
            this.batchSize = batchSize
        }

        if (shuffle) {
            shuffleMatch(inputs, targets)
        }

        let error = 0
        let output = await this.predictOnBatch(inputs)
        let neuronErrors: Tensor[] = []
        for (let index = 0; index < batchSize; index++) {
            error += this.lossFunction.forward(output[index], targets[index])
            neuronErrors[index] = this.lossFunction.backward(output[index], targets[index])

        }
        for (let i = this.numLayers - 1; i >= 0; i--) {
            neuronErrors = this.layers[i].backward(neuronErrors)
        }
        // Update error, weights and biases for the current batch
        this.batch++
        error /= batchSize
        this.error = error
        
        this.update(this.batch)
    }
    update(index: number) {
        for (let i = 0; i < this.trainableLayers.length; i++) {
            this.trainableLayers[i].update(this.optimizer, this.batchSize, index)
        }
    }
    
    mutateRandomLayer(mutateBias: boolean = false, mutationRate: number = 0.1, mutation: number = 0.1) {
        let index = Math.floor(Math.random() * this.trainableLayers.length)

        let mutationRange = [-mutation, mutation]
        this.trainableLayers[index].mutateWeights(mutationRate, mutationRange)
        if (mutateBias == false) return
        this.trainableLayers[index].mutateBias(mutationRate, mutationRange)
    }
    mutate(mutateBias: boolean = false, mutationRate: number = 0.1, mutation: number = 0.5) {
        let mutationRange = [-mutation, mutation]

        for (let index = 0; index < this.trainableLayers.length; index++) {
            this.trainableLayers[index].mutateWeights(mutationRate, mutationRange)
            if (mutateBias == false) continue
            this.trainableLayers[index].mutateBias(mutationRate, mutationRange)
        }
    }

    save(){
        let layers = []
        for (let i = 0; i < this.layers.length; i++) {
            layers.push(this.layers[i].save())
        }
        return JSON.stringify({
            loss: this.loss,
            optimizer: this.optimizer,
            layers
        })
    }
    static loadModel(modelSerialized: any) {
        let newModel = new Sequential()
        for (let index = 0; index < modelSerialized.layers.length; index++) {
            let newLayer: LayerClasses
            if (modelSerialized.layers[index].type == 'activation') {
                newLayer = activationLayer(modelSerialized.layers[index].name)
                newModel.add(newLayer)
            } else {
                newLayer = layerFromObject(modelSerialized.layers[index])
                newModel.add(newLayer, true)
            }
        }
        newModel.compile({
            loss: modelSerialized.loss,
            optimizer: optimizersHandler(modelSerialized.optimizer)
        })

        return newModel
    }
    copy() {
        return Sequential.loadModel(JSON.parse(this.save()))
    }
    static copy(model: Sequential) {
        return Sequential.loadModel(JSON.parse(model.save()))
    }
}
