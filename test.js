const {Variable, Operation} = require("./core")
const Arr = require("./Arr")

function f(x) {
    let y = x.pow(4).minus(x.pow(2).mul(2))
    return y
}

let x = new Variable(Arr(2))
let iters = 10

for(let i = 0; i < iters; i++) {
    console.log(i, x.view)

    let y = f(x)
    x.name = "x"
    x.cleargrad()
    y.backward(false, true)

    let gx = x.grad
    x.cleargrad()
    gx.backward(true)
    gx2 = x.grad

    x.data = x.data.minus(gx.data.div(gx2.data))
}
console.log(x.data)