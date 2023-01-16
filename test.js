const {Variable, no_grad} = require("./core")
const Arr = require("./Arr")

function sphere(x, y) {
    let z = x.pow(2).add(y.pow(2))
    return z
}

function matyas(x, y) {
    let z = sphere(x, y).mul(0.26).sub(x.mul(y).mul(0.48))
    return z
}

let x = new Variable(Arr(1))
let y = new Variable(Arr(1))
let z = matyas(x, y)
z.backward()
console.log(x.grad, y.grad)