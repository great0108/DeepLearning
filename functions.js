(function() {
    "use strict"
    const Arr = require("./Arr")
    const {Operation, Variable, as_variable, as_array} = require("./core")

    function Sin() {
        return Operation.inherit(Sin)
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
        return Operation.inherit(Cos)
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


    function Tanh() {
        return Operation.inherit(Tanh)
    }

    Tanh.prototype.__proto__ = Operation.prototype

    Tanh.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.tanh(v))
        return y
    }

    Tanh.prototype.backward = function(gy) {
        let y = this.outputs[0]
        let gx = gy.mul(y.pow(2).rminus(1))
        return gx
    }


    function Exp() {
        return Operation.inherit(Exp)
    }

    Exp.prototype.__proto__ = Operation.prototype

    Exp.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.exp(v))
        return y
    }

    Exp.prototype.backward = function(gy) {
        let y = this.outputs[0]
        let gx = gy.mul(y)
        return gx
    }


    function Log() {
        return Operation.inherit(Log)
    }

    Log.prototype.__proto__ = Operation.prototype

    Log.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.log(v))
        return y
    }

    Log.prototype.backward = function(gy) {
        let x = this.inputs[0]
        let gx = gy.div(x)
        return gx
    }


    function Reshape(shape) {
        let result = Operation.inherit(Reshape)
        result.shape = shape
        return result
    }

    Reshape.prototype.__proto__ = Operation.prototype

    Reshape.prototype.forward = function(x) {
        this.x_shape = x.shape
        let y = x.reshape(this.shape)
        return y
    }

    Reshape.prototype.backward = function(gy) {
        return reshape(gy, this.x_shape)
    }


    function Transpose(axes) {
        let result = Operation.inherit(Transpose)
        result.axes = axes
        return result
    }

    Transpose.prototype.__proto__ = Operation.prototype

    Transpose.prototype.forward = function(x) {
        let y = x.a
    }


    function Sum(axis, keepdims) {
        let result = Operation.inherit(Sum)
        result.axis = axis
        result.keepdims = keepdims
        return result
    }

    Sum.prototype.__proto__ = Operation.prototype

    Sum.prototype.forward = function(x) {
        this.x_shape = x.shape
        let y = x.sum(axis=this.axis, keepdims=this.keepdims)
        return y
    }


    function sin(x) {
        return Sin()(x)
    }

    function cos(x) {
        return Cos()(x)
    }

    function tanh(x) {
        return Tanh()(x)
    }

    function exp(x) {
        return Exp()(x)
    }

    function log(x) {
        return Log()(x)
    }

    function reshape(x, shape) {
        if(x.shape == shape) {
            return as_variable(x)
        }
        return Reshape(shape)(x)
    }


    module.exports = {
        sin : sin,
        cos : cos,
        tanh : tanh,
        exp : exp,
        log : log
    }
})()