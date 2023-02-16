const {Variable, no_grad, test_mode} = require("./core")
const {sigmoid, mean_squared_error, softmax_cross_entropy, accuracy, dropout} = require("./functions")
const {Layer, Linear} = require("./layers")
const {MLP} = require("./models")
const {SGD, MomentumSGD, AdaGrad, AdaDelta, Adam} = require("./optimizers")
const Arr = require("./Arr")
const {Spiral, Mnist} = require("./datasets")
const {DataLoader} = require("./dataloaders")
const utils = require("./utils")
const setting = require("./setting")



let max_epoch = 100
let batch_size = 20
let hidden_size = 20
let lr = 0.1

let train_set = new Spiral()
let train_loader = new DataLoader(train_set, batch_size)

let model = MLP([hidden_size, 3])
let optimizer = Adam(lr).setup(model)

for(let epoch = 0; epoch < max_epoch; epoch++) {
    let sum_loss = 0
    let sum_acc = 0
    for(let data of train_loader) {
        // 미니배치
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

    if((epoch+1) % 20 === 0) {
        let avg_loss = sum_loss / train_set.length
        let avg_acc = sum_acc / train_set.length
        console.log("epoch : " + (epoch+1) + "  loss : " + avg_loss + "  accuracy : " + avg_acc)
        // epoch : 20  loss : 0.10584555196609018  accuracy : 0.98
        // epoch : 40  loss : 0.04696663155208051  accuracy : 0.9833333333333333
        // epoch : 60  loss : 0.031889515063875586  accuracy : 0.99
        // epoch : 80  loss : 0.024977741160538414  accuracy : 0.9933333333333333
        // epoch : 100  loss : 0.02156269388124971  accuracy : 0.9933333333333333
    }
}