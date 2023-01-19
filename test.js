const {Variable, Operation} = require("./core")
const {sin} = require("./functions")
const Arr = require("./Arr")

function f(x) {
    let y = x.pow(4).minus(x.pow(2).mul(2))
    return y
}

let x = new Variable(Arr.range(-3, 3, 0.5))
let y = sin(x)
y.backward(true)

let logs = [y.data]

for(let i = 0; i < 3; i++) {
    console.log(x.grad)
    logs.push(x.grad.data)
    let gx = x.grad
    x.cleargrad()
    gx.backward(true)
}

for(let log of logs) {
    console.log(log)
}