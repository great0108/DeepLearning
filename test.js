const {Variable, sum} = require("./core")
const {matmul} = require("./functions")
const Arr = require("./Arr")

let x = Arr.zeros(100, 1).deepMap(v => Math.random())
let y = x.mul(2).plus(5).deepMap(v => v + Math.random())

x = new Variable(x)
y = new Variable(y)

let W = new Variable(Arr.zeros(1, 1))
let b = new Variable(Arr.zeros(1))

function predict(x) {
    let y = matmul(x, W).plus(b)
    return y
}

function mean_squared_error(x0, x1) {
    let diff = x0.minus(x1)
    return sum(diff.pow(2)).div(diff.length)
}

let lr = 0.1
let iters = 30

for(let i = 0; i < iters; i++) {
    let y_pred = predict(x)
    let loss = mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data = W.data.minus(W.grad.data.mul(lr))
    b.data = b.data.minus(b.grad.data.mul(lr))
    console.log(W.view, b.view, loss.view)
}