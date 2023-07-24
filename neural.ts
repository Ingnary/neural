declare type Iterableify<T> = { [K in keyof T]: Iterable<T[K]> }

function* zip<T extends Array<any>>(
    ...toZip: Iterableify<T>
): Generator<T> {
    // Get iterators for all of the iterables.
    const iterators = toZip.map(iterable => iterable[Symbol.iterator]())

    while (true) {
        // Advances all of the iterators.
        const results = iterators.map(iterable => iterable.next())

        // If any of the iterators are done, we should stop.
        if (results.some(({ done }) => done)) {
            break
        }

        // We can assert the yield type, since we knoe none
        // of the iterators are done.
        yield results.map(({ value }) => value) as T
    }
}

function* range(stop: number) {
    let index = 0
    while (index < stop)
        yield index++
}

declare type InitOptions = { learningRate: number, activateFunc: (v: number) => number, activateFunc_d: (v: number) => number }
declare type TrainOptions = { times: number, logTimes?: number }

class Neural {
    layers: number[][]
    grads: number[][]
    weights: number[][][]

    learningRate: number
    activateFunc: (v: number) => number
    activateFunc_d: (v: number) => number

    constructor(eachLayerSize: number[], { learningRate, activateFunc, activateFunc_d }: InitOptions) {
        this.layers = eachLayerSize.map(size => new Array<number>(size).fill(0))
        this.grads = eachLayerSize.map(size => new Array<number>(size).fill(0))
        this.weights = Array.from(zip(eachLayerSize.slice(0, -1), eachLayerSize.slice(1))).map(([leftSize, rightSize]) => new Array<number[]>(rightSize).fill(new Array<number>(leftSize).fill(0)))
        this.weights.forEach(weight => weight.forEach(wr => wr.forEach((_, i, wr) => wr[i] = 1 - Math.random() * 2)))
        this.learningRate = learningRate
        this.activateFunc = activateFunc
        this.activateFunc_d = activateFunc_d
    }

    Forward(inputData: number[]): void {
        this.layers.forEach(layer => layer.fill(0))
        Object.assign(this.layers[0], inputData)
        for (const [layerLeft, layerRight, weight] of zip(this.layers, this.layers.slice(1), this.weights)) {
            for (const i of range(layerRight.length)) {
                for (const [w, ll] of zip(weight[i], layerLeft)) { layerRight[i] += w * ll }
                layerRight[i] = this.activateFunc(layerRight[i])
            }
        }
    }

    Backward(idealOutput: number[]): void {
        this.grads.forEach(grad => grad.fill(0))
        for (const i of range(this.layers.at(-1).length)) {
            for (const [real, ideal] of zip(this.layers.at(-1), idealOutput)) { this.grads.at(-1)[i] = 2 / this.layers.at(-1).length * (real - ideal) }
        }
        this.layers.reverse(), this.grads.reverse(), this.weights.reverse()
        for (const [layerLeft, layerRight, gradLeft, gradRight, weight] of zip(this.layers.slice(1), this.layers, this.grads.slice(1), this.grads, this.weights)) {
            const transWeight = zip(...weight)
            for (const i of range(gradLeft.length)) {
                for (const [lr, gr, w] of zip(layerRight, gradRight, transWeight.next().value as number[])) { gradLeft[i] += this.activateFunc_d(lr) * gr * w }
            }
            for (const [i, lr, gr] of zip(range(gradRight.length), layerRight, gradRight)) {
                for (const [j, ll] of zip(range(weight[i].length), layerLeft)) { weight[i][j] -= this.learningRate * ll * this.activateFunc_d(lr) * gr }
            }
        }
        this.layers.reverse(), this.grads.reverse(), this.weights.reverse()
    }

    Train(data: [number[], number[]][], { times, logTimes = 1 }: TrainOptions): void {
        const gap = logTimes <= 0 ? times : logTimes > times ? 1 : Math.floor(times / logTimes)
        for (const i of range(times)) {
            for (const [inputData, idealOutput] of data) {
                this.Forward(inputData)
                this.Backward(idealOutput)
            }
            if ((i + 1) % gap == 0)
                console.log(`loss: ${this.grads.at(-1).reduce((sum, value) => sum + value ** 2, 0) ** 0.5}, weights: ${this.weights}`)
        }
    }
}

function targetFunc(x: number): number {
    return 1 * x ** 2 + 2 * x + 3
}

const n = new Neural([3, 1], { learningRate: 0.01, activateFunc: (v: number) => v, activateFunc_d: (v: number) => 1 })
n.Train([-2, -1, 0, 1, 2].map(x => [[x ** 2, x, 1], [targetFunc(x)]]), { times: 100, logTimes: 5 })