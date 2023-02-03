const {Variable} = require("./core")
const {sigmoid, mean_squared_error, softmax_cross_entropy} = require("./functions")
const {Layer, Linear} = require("./layers")
const {MLP} = require("./models")
const {SGD, MomentumSGD, AdaGrad, AdaDelta, Adam} = require("./optimizers")
const Arr = require("./Arr")
const {Spiral} = require("./datasets")

let max_epoch = 100
let batch_size = 20
let hidden_size = 20
let lr = 0.2

let train_set = new Spiral()
let model = MLP([hidden_size, 3])
let optimizer = Adam(lr).setup(model)

let data_size = train_set.length
let max_iter = Math.ceil(data_size / batch_size)

for(let epoch = 0; epoch < max_epoch; epoch++) {
    sum_loss = 0
    for(let i = 0; i < max_iter; i++) {
        let batch = train_set.slice(i * batch_size, (i+1) * batch_size)
        let batch_x = batch[0]
        let batch_t = batch[1]

        let y = model(batch_x)
        let loss = softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += loss.data[0] * batch_x.length
    }

    let avg_loss = sum_loss / data_size
    if((epoch+1) % 20 === 0) {
        console.log("epoch : " + (epoch+1) + "  loss : " + avg_loss)
    }
}