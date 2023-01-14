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
        return x.map(v => Math.pow(v, 2))
    }


    let x = new Variable(Arr(10))
    console.log(x.data)
    let f = new Square()
    let y = f.cal(x)
    console.log(y.constructor)
    console.log(y.data)

})()