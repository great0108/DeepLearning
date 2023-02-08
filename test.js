const {Variable, no_grad} = require("./core")
const {sigmoid, mean_squared_error, softmax_cross_entropy, accuracy} = require("./functions")
const {Layer, Linear} = require("./layers")
const {MLP} = require("./models")
const {SGD, MomentumSGD, AdaGrad, AdaDelta, Adam} = require("./optimizers")
const Arr = require("./Arr")
const {Spiral, Mnist} = require("./datasets")
const {DataLoader} = require("./dataloaders")
const utils = require("./utils")
const setting = require("./setting")


let max_epoch = 5
let batch_size = 200
let hidden_size = 100
let lr = 0.01

let train_set = new Mnist(true)
let test_set = new Mnist(false)
let train_loader = new DataLoader(train_set, batch_size)
let test_loader = new DataLoader(test_set, batch_size, false)

let model = MLP([hidden_size, 10])
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

        sum_loss += loss.data[0] * t.length
        sum_acc += acc.data[0] * t.length
    }

    if((epoch+1) % 1 === 0) {
        let avg_loss = sum_loss / train_set.length
        let avg_acc = sum_acc / train_set.length

        let test_loss = 0
        let test_acc = 0
        no_grad(() => {
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
        console.log("epoch : " + (epoch+1) + "  loss : " + avg_loss + "  accuracy : " + avg_acc + "  test loss : " + avg_loss2 + "  test accuracy : " + avg_acc2)
    }
}
model.save_weights("test.json")
