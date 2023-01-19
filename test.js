const {Variable, Operation} = require("./core")
const Arr = require("./Arr")

function Sin() {
    let result = Operation.call(this)
    Object.setPrototypeOf(result, Sin.prototype)
    return result
}

Sin.prototype.__proto__ = Operation.prototype

Sin.prototype.forward = function(x) {
    let y = x.deepMap(v => Math.sin(v))
    return y
}

Sin.prototype.backward = function(gy) {
    let x = this.inputs[0].data
    let gx = x.deepMap(v => Math.cos(v)).mul(gy)
    return gx
}

function sin(x) {
    return new Sin()(x)
}

let x = new Variable(Arr(Math.PI / 4))
let y = sin(x)
y.backward()
console.log(y.data)
console.log(x.grad)

function factorial(a) {
    let result = 1
    for(let i = a; i > 1; i--) {
        result *= i
    }
    return result
}

function my_sin(x, threshold) {
    threshold = threshold === undefined ? 0.0001 : threshold
    let y = 0
    for(let i = 0; i < 100000; i++) {
        let c = Math.pow(-1, i) / factorial(2 * i + 1)
        let t = x.pow(2 * i + 1).mul(c)
        y = t.add(y)
        if(Math.abs(t.data) < threshold) {
            break
        }
    }
    return y
}

x = new Variable(Arr(Math.PI / 4))
y = my_sin(x, 1e-10)
y.backward()
console.log(y.data)
console.log(x.grad)