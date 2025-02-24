const {Variable, no_grad, test_mode} = require("./core")
const {sigmoid, mean_squared_error, softmax_cross_entropy, accuracy, dropout, relu, pooling, average_pooling, flatten} = require("./functions")
const {Layer, Linear, Conv2d, RNN, LSTM} = require("./layers")
const {MLP} = require("./models")
const {SGD, MomentumSGD, AdaGrad, AdaDelta, Adam} = require("./optimizers")
const Arr = require("./Arr")
const {Spiral, Mnist, SinCurve} = require("./datasets")
const {DataLoader, SeqDataLoader} = require("./dataloaders")

let max_epoch = 100
let batch_size = 30
let hidden_size = 100
let bptt_length = 30
let lr = 0.001

function BetterRNN(hidden_size, out_size) {
    this.rnn = LSTM(hidden_size)
    this.fc = Linear(out_size)
    return this.make(this)
}

BetterRNN.prototype.__proto__ = Layer.prototype

BetterRNN.prototype.reset_state = function() {
    this.rnn.reset_state()
}

BetterRNN.prototype.forward = function(x) {
    let h = this.rnn(x)
    let y = this.fc(h)
    return y
}

let train_set = SinCurve(true)
let test_set = SinCurve(false)
let train_loader = SeqDataLoader(train_set, batch_size)
let test_loader = SeqDataLoader(test_set, batch_size)
let seq_len = train_set.length

let model = new BetterRNN(hidden_size, 1)
optimizer = Adam(lr).setup(model)
console.log("train start")

for(let epoch = 0; epoch < max_epoch; epoch++) {
    model.reset_state()
    let sum_loss = 0
    let loss = 0
    let count = 0

    for(let data of train_loader) {
        let [x, t] = data

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
    if((epoch+1) % 10 === 0) {
        console.log("epoch : " + (epoch+1) + "  loss : " + avg_loss)
        sum_loss = 0
        no_grad(() => {
            for(let data of test_loader) {
                let [x, t] = data

                let y = model(x)
                let loss = mean_squared_error(y, t)
                sum_loss += loss.data[0]
            }
        })
        avg_loss = sum_loss / test_set.length
        console.log("test loss : " + avg_loss)
    }
}

let xs = Arr.range(0, 2 * Math.PI + 1e-6, 2 * Math.PI / (1000-1))
xs = Arr(xs.map(v => Math.sin(v)).slice(0, 100)).expand(1)

model.reset_state()
pred = []
no_grad(() => {
    for(let x of xs) {
        let y = model(x.reshape(1, 1))
        pred.push(y.data[0])
    }
})
console.log(pred)