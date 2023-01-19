const {Variable, Operation} = require("./core")
const Arr = require("./Arr")

function f(x) {
    let y = x.pow(4).minus(x.pow(2).mul(2))
    return y
}

function gx2(x) {
    return x.pow(2).mul(12).minus(4)
}

let x = new Variable(Arr(2))
let iters = 10

for(let i = 0; i < iters; i++) {
    console.log(i, x.view)

    let y = f(x)
    x.cleargrad()
    y.backward()

    x.data = x.data.minus(x.grad.div(gx2(x.data)))
}
console.log(x.data)