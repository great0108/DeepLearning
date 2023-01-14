(function() {
    "use strict"
    const Arr = require("./Arr")

    function Variable(data) {
        this.data = data
        this.grad = null
    }

    function Function() {
        let result = function() {
            return result.__call__.apply(result, arguments);
        }
        result.__proto__ = Function.prototype;
        return result;
    }

    Function.prototype.__call__ = function(input) {
        let x = input.data
        let y = this.forward(x)
        let output = new Variable(y)
        this.input = input
        return output
    }

    Function.prototype.forward = function() {
        throw new Error("NotImplemented")
    }

    Function.prototype.backward = function() {
        throw new Error("NotImplemented")
    }


    function Square() {
        let result = Function.call(this)
        result.__proto__ = Square.prototype
        return result
    }

    Square.prototype.__proto__ = Function.prototype

    Square.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.pow(v, 2))
        return y
    }

    Square.prototype.backward = function(gy) {
        let x = this.input.data
        let gx = x.mul(2).mul(gy)
        return gx
    }


    function Exp() {
        let result = Function.call(this)
        result.__proto__ = Exp.prototype
        return result
    }

    Exp.prototype.__proto__ = Function.prototype

    Exp.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.exp(v))
        return y
    }

    Exp.prototype.backward = function(gy) {
        let x = this.input.data
        let gx = x.deepMap(v => Math.exp(v)).mul(gy)
        return gx
    }

    let A = new Square()
    let B = new Exp()
    let C = new Square()

    let x = new Variable(Arr(0.5))
    let a = A(x)
    let b = B(a)
    let y = C(b)

    y.grad = Arr(1)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    console.log(x.grad)
    console.log(A)

})()