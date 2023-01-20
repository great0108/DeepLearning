const {Variable, Operation} = require("./core")
const {sin, tanh} = require("./functions")
const Arr = require("./Arr")

let x = new Variable(Arr(1))
let y = tanh(x)
y.backward(true)

iters = 1
for(let i = 0; i < iters; i++) {
    let gx = x.grad
    x.cleargrad()
    gx.backward(true)
}

gx = x.grad
console.log(x.grad)