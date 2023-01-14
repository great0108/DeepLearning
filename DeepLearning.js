(function() {
    "use strict"
    const Arr = require("./Arr")

    function Callable() {
        let self = this
        let result = function Callable() {
            return self.__call__.apply(self, arguments);
        }
        result.__proto__ = Callable.prototype;
        return result;
    }
    Callable.prototype.__call__ = function(x) {
        return x+1
    }
    Callable.prototype.a = function() {
        return 1
    }
    
    function Callable2() {
        let result = Callable.call(this)
        result.__proto__ = Callable.prototype;
        return result;
    }
    Callable2.prototype.__proto__ = Callable.prototype

    function Variable(data) {
        this.data = data
    }

    function Function() {
        let self = this
        let result = function() {
            return self.__call__.apply(self, arguments);
        }
        result.__proto__ = Function.prototype;
        return result;
    }
    Function.prototype.__call__ = function(input) {
        let x = input.data
        let y = this.forward(x)
        let output = new Variable(y)
        this.input = input
        this.output = output
        return output
    }
    Function.prototype.forward = function() {
        throw new Error("NotImplemented")
    }

    function Square() {
        let result = Function.call(this)
        result.__proto__ = Square.prototype
        return result
    }
    Square.prototype.__proto__ = Function.prototype
    Square.prototype.forward = function(x) {
        return x.deepMap(v => Math.pow(v, 2))
    }

    function Exp() {
        let result = Function.call(this)
        result.__proto__ = Exp.prototype
        return result
    }
    Exp.prototype.__proto__ = Function.prototype
    Exp.prototype.forward = function(x) {
        return x.deepMap(v => Math.exp(v))
    }

    function numerical_diff(f, x, eps=1e-4) {
        let x0 = new Variable(x.data.minus(eps))
        let x1 = new Variable(x.data.plus(eps))
        let y0 = f(x0)
        let y1 = f(x1)
        return y1.data.minus(y0.data).div(2 * eps)
    }


    let f = new Square()
    let x = new Variable(Arr(2))
    let dy = numerical_diff(f, x)
    console.log(dy)

    f = function (x) {
        let A = new Square()
        let B = new Exp()
        let C = new Square()
        return C(B(A(x)))
    }

    x = new Variable(Arr(0.5))
    dy = numerical_diff(f, x)
    console.log(dy)

})()