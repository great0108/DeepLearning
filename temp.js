function Square() {
    let result = Operation.call(this)
    Object.setPrototypeOf(result, Square.prototype)
    return result
}

Square.prototype.__proto__ = Operation.prototype

Square.prototype.forward = function(x) {
    let y = x.deepMap(v => Math.pow(v, 2))
    return y
}

Square.prototype.backward = function(gy) {
    let x = this.inputs[0].data
    let gx = x.mul(2).mul(gy)
    return gx
}


function Exp() {
    let result = Operation.call(this)
    Object.setPrototypeOf(result, Exp.prototype)
    return result
}

Exp.prototype.__proto__ = Operation.prototype

Exp.prototype.forward = function(x) {
    let y = x.deepMap(v => Math.exp(v))
    return y
}

Exp.prototype.backward = function(gy) {
    let x = this.inputs[0].data
    let gx = x.deepMap(v => Math.exp(v)).mul(gy)
    return gx
}

function square(x) {
    return new Square()(x)
}

function exp(x) {
    return new Exp()(x)
}

function numerical_diff(f, x, eps=1e-4) {
    let x0 = new Variable(x.data.minus(eps))
    let x1 = new Variable(x.data.plus(eps))
    let y0 = f(x0)
    let y1 = f(x1)
    return y1.data.minus(y0.data).div(2 * eps)
}