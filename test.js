const {Variable, no_grad, test_mode} = require("./core")
const {sigmoid, mean_squared_error, softmax_cross_entropy, accuracy, dropout, relu, polling, flatten} = require("./functions")
const {Layer, Linear, Conv2d} = require("./layers")
const {MLP} = require("./models")
const {SGD, MomentumSGD, AdaGrad, AdaDelta, Adam} = require("./optimizers")
const Arr = require("./Arr")
const {Spiral, Mnist} = require("./datasets")
const {DataLoader} = require("./dataloaders")
const utils = require("./utils")
const setting = require("./setting")

function ConvModel() {
    this.conv1 = Conv2d(8, 3)
    this.conv2 = Conv2d(8, 3)
    this.conv3 = Conv2d(16, 3)
    this.l1 = Linear(100)
    this.l2 = Linear(10)
    return this.make(this)
}
ConvModel.prototype.__proto__ = Layer.prototype

ConvModel.prototype.forward = function(x) {
    let y = relu(this.conv1(x))
    y = relu(this.conv2(y))
    y = polling(y, 2, 2)
    y = relu(this.conv3(y))
    y = flatten(y)
    y = relu(this.l1(y))
    y = this.l2(y)
    return y
}

let max_epoch = 10
let batch_size = 200
let lr = 0.01

let train_set = Mnist(true, a => a.deepMap(v => Number(v) / 255).reshape(-1, 1, 28, 28))
let train_loader = DataLoader(train_set, batch_size)

let model = new ConvModel()
let optimizer = Adam(lr).setup(model)
console.log("start")
for(let epoch = 0; epoch < max_epoch; epoch++) {
    let sum_loss = 0
    let sum_acc = 0
    let time = Date.now()
    for(let data of train_loader) {
        let [x, t] = data
        console.log(x.shape, t.shape)

        let y = model(x)
        let loss = softmax_cross_entropy(y, t)
        let acc = accuracy(y, t)
        model.cleargrads()
        console.log(Date.now() - time)
        loss.backward()
        optimizer.update()

        sum_loss += loss.data[0] * t.length
        sum_acc += acc.data[0] * t.length
        console.log(Date.now() - time)
    }

    if((epoch+1) % 1 === 0) {
        let avg_loss = sum_loss / train_set.length
        let avg_acc = sum_acc / train_set.length
        console.log("epoch : " + (epoch+1) + "  loss : " + avg_loss + "  accuracy : " + avg_acc)
    }
}