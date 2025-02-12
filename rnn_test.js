const {Variable, no_grad, test_mode} = require("./core")
const {sigmoid, mean_squared_error, softmax_cross_entropy, accuracy, dropout, relu, pooling, average_pooling, flatten} = require("./functions")
const {Layer, Linear, Conv2d, RNN} = require("./layers")
const {MLP} = require("./models")
const {SGD, MomentumSGD, AdaGrad, AdaDelta, Adam} = require("./optimizers")
const Arr = require("./Arr")
const {Spiral, Mnist, SinCurve} = require("./datasets")
const {DataLoader} = require("./dataloaders")

let max_epoch = 100
let hidden_size = 100
let bptt_length = 30
let lr = 0.001

function SimpleRNN(hidden_size, out_size) {
    this.rnn = RNN(hidden_size)
    this.fc = Linear(out_size)
    return this.make(this)
}

SimpleRNN.prototype.__proto__ = Layer.prototype

SimpleRNN.prototype.reset_state = function() {
    this.rnn.reset_state()
}

SimpleRNN.prototype.forward = function(x) {
    let h = this.rnn(x)
    let y = this.fc(h)
    return y
}


let train_set = SinCurve(true)
let test_set = SinCurve(false)
let seq_len = train_set.length

let model = new SimpleRNN(hidden_size, 1)
optimizer = Adam(lr).setup(model)
console.log("train start")

for(let epoch = 0; epoch < max_epoch; epoch++) {
    model.reset_state()
    let sum_loss = 0
    let loss = 0
    let count = 0

    for(let data of train_set) {
        // console.log(data)
        let [x, t] = data
        x = x.reshape(1, 1)

        let y = model(x)
        loss = mean_squared_error(y, t).plus(loss)
        count += 1

        if(count % bptt_length == 0 || count == seq_len) {
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

            sum_loss += loss.data[0]
            loss = 0
        }
    }

    let avg_loss = sum_loss / train_set.length
    if((epoch+1) % 20 === 0) {
        console.log("epoch : " + (epoch+1) + "  loss : " + avg_loss)
        sum_loss = 0
        no_grad(() => {
            for(let data of test_set) {
                let [x, t] = data
                x = x.reshape(1, 1)

                let y = model(x)
                let loss = mean_squared_error(y, t)
                sum_loss += loss.data[0]
            }
        })
        avg_loss = sum_loss / test_set.length
        console.log("test loss : " + avg_loss)
    }
}
model.save_weights("mlp.json")