export const calculateFans = (shape: number[]) => {
    shape.map(x => Math.floor(x))
    let fanIn = 1, fanOut = 1
    if (shape.length == 2) {
        fanIn = shape[1]
        fanOut = shape[0]
    } else if (shape.length == 4) {
        // Shape = #Filters, #Channels, kernelRows, kernelCols
        let receptiveField = shape[2] * shape[3]
        fanIn = receptiveField * shape[1]
        fanOut = receptiveField * shape[0]
    }
    return [fanIn, fanOut]
}