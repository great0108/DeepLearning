const {Variable, Operation} = require("./core")
const Arr = require("./Arr")

function rosenbrock(x0, x1) {
    let y = x1.minus(x0.pow(2)).pow(2).mul(100).plus(x0.minus(1).pow(2))
    return y
}

let x0 = new Variable(Arr(0))
let x1 = new Variable(Arr(2))
let lr = 0.001
let iters = 1000

for(let i = 0; i < iters; i++) {
    let y = rosenbrock(x0, x1)
    console.log(x0.view, x1.view)
    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data = x0.data.minus(x0.grad.mul(lr))
    x1.data = x1.data.minus(x1.grad.mul(lr))
}
console.log(x0.data)
console.log(x1.data)