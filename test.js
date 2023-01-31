const {Variable} = require("./core")
const {sigmoid, mean_squared_error} = require("./functions")
const {Layer, Linear} = require("./layers")
const {MLP} = require("./models")
const Arr = require("./Arr")

let x = Arr.rand(100, 1)
let y = x.deepMap(v => Math.sin(2 * Math.PI * v) + Math.random())

let model = MLP([20, 10, 1])
let lr = 0.2
let iters = 1000

for(let i = 0; i < iters; i++) {
    let y_pred = model(x)
    let loss = mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for(let param of model.params()) {
        param.data = param.data.minus(param.grad.data.mul(lr))
    }

    if((i+1) % 100 === 0) {
        console.log(loss.view)
    }
}