(function() {
    "use strict"
    const Arr = require("./Arr")

    function Variable(data) {
        if(data !== null) {
            if(!(data instanceof Arr)) {
                throw new Error(data.constructor.name + " is not supported")
            }
        }

        this.data = data
        this.grad = null
        this.creator = null
    }

    Variable.prototype.set_creator = function(func) {
        this.creator = func
    }

    Variable.prototype.backward = function() {
        if(this.grad === null) {
            this.grad = Arr.fill(this.data.shape(), 1)
        }

        let funcs = [this.creator]
        while(funcs.length > 0) {
            let f = funcs.pop()
            let x = f.input, y = f.output
            x.grad = f.backward(y.grad)

            if(x.creator !== null) {
                funcs.push(x.creator)
            }
        }
    }

    function as_array(x) {
        if(!isNaN(x)) {
            return Arr(x)
        }
        return x
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
        let y = this.forward(as_array(x))
        let output = new Variable(y)
        output.set_creator(this)
        this.input = input
        this.output = output
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

    let x = new Variable(Arr(Math.random()))
    let y = square(x)
    y.backward()
    let num_grad = numerical_diff(square, x)
    console.log(x.grad, num_grad)

})()