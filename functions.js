(function() {
    "use strict"
    const Arr = require("./Arr")
    const {Operation, Variable, as_variable, as_array} = require("./core")

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
        let x = this.inputs[0]
        let gx = cos(x).mul(gy)
        return gx
    }


    function Cos() {
        let result = Operation.call(this)
        Object.setPrototypeOf(result, Cos.prototype)
        return result
    }

    Cos.prototype.__proto__ = Operation.prototype

    Cos.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.cos(v))
        return y
    }

    Cos.prototype.backward = function(gy) {
        let x = this.inputs[0]
        let gx = sin(x).mul(-1).mul(gy)
        return gx
    }

    
    function sin(x) {
        return new Sin()(x)
    }

    function cos(x) {
        return new Cos()(x)
    }



    module.exports = {
        sin : sin,
        cos : cos
    }
})()