(function() {
    "use strict"
    const Arr = require("./Arr")

    function Variable(data) {
        this.data = data
    }

    function Function() {}
    Function.prototype.cal = function(input) {
        let x = input.data
        let y = this.forward(x)
        let output = new Variable(y)
        return output
    }
    Function.prototype.forward = function() {
        throw new Error("NotImplemented")
    }

    function Square() {}
    Square.prototype.__proto__ = Function.prototype
    Square.prototype.forward = function(x) {
        return x.deepMap(v => Math.pow(v, 2))
    }

    function Exp() {}
    Exp.prototype.__proto__ = Function.prototype
    Exp.prototype.forward = function(x) {
        return x.deepMap(v => Math.exp(v))
    }


    let A = new Square()
    let B = new Exp()
    let C = new Square()

    let x = new Variable(Arr(0.5))
    let a = A.cal(x)
    console.log(a)
    let b = B.cal(a)
    console.log(b)
    let y = C.cal(b)
    console.log(y.data)

})()