const {test_mode} = require("../core")
const {softmax_cross_entropy, accuracy, relu, average_pooling, flatten} = require("../functions")
const {Layer, Linear, Conv2d} = require("../layers")
const {Adam} = require("../optimizers")
const {Mnist} = require("../datasets")
const {DataLoader} = require("../dataloaders")

function ConvModel() {
    this.conv1 = Conv2d(16, 3, 1, 1)
    this.l1 = Linear(100)
    this.l2 = Linear(10)
    return this.make(this)
}
ConvModel.prototype.__proto__ = Layer.prototype

ConvModel.prototype.forward = function(x) {
    let y = relu(this.conv1(x))
    y = average_pooling(y, 2, 2)
    y = flatten(y)
    y = relu(this.l1(y))
    y = this.l2(y)
    return y
}

let max_epoch = 10
let batch_size = 10
let lr = 0.01

let train_set = Mnist(true, a => a.deepMap(v => Number(v) / 255).reshape(-1, 1, 28, 28), null, 1000)
let test_set = Mnist(false, a => a.deepMap(v => Number(v) / 255).reshape(-1, 1, 28, 28), null, 1000)
let train_loader = DataLoader(train_set, batch_size)
let test_loader = DataLoader(test_set, batch_size)

let model = new ConvModel()
let optimizer = Adam(lr).setup(model)
console.log("train start")

for(let epoch = 0; epoch < max_epoch; epoch++) {
    let sum_loss = 0
    let sum_acc = 0
    for(let data of train_loader) {
        let [x, t] = data
        let y = model(x)
        let loss = softmax_cross_entropy(y, t)
        let acc = accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += loss.data[0] * x.length
        sum_acc += acc.data[0] * x.length
    }

    if((epoch+1) % 1 === 0) {
        let avg_loss = sum_loss / train_set.length
        let avg_acc = sum_acc / train_set.length
        let test_loss = 0
        let test_acc = 0
        test_mode(() => {
            for(let data of test_loader) {
                let [x, t] = data
                let y = model(x)
                let loss = softmax_cross_entropy(y, t)
                let acc = accuracy(y, t)
                test_loss += loss.data[0] * t.length
                test_acc += acc.data[0] * t.length
            }
        })

        let avg_loss2 = test_loss / test_set.length
        let avg_acc2 = test_acc / test_set.length
        console.log("epoch : " + (epoch+1) + "  loss : " + avg_loss + "  accuracy : " + avg_acc +
                    "  test loss : " + avg_loss2 + "  test accuracy : " + avg_acc2)
    }
}