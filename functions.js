(function() {
    "use strict"
    const Arr = require("./Arr")
    const {Operation, Variable, as_variable, as_array} = require("./core")
    const {reshape_sum_backward} = require("./utils")

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


    function Sum(axis, keepdims) {
        let result = Operation.inherit(Sum)
        result.axis = axis
        result.keepdims = keepdims
        return result
    }

    Sum.prototype.__proto__ = Operation.prototype

    Sum.prototype.forward = function(x) {
        this.x_shape = x.shape
        let y = x.sum(this.axis, this.keepdims)
        return y
    }

    Sum.prototype.backward = function(gy) {
        gy = reshape_sum_backward(gy, this.x_shape, this.axis, this.keepdims)
        let gx = broadcast_to(gy, this.x_shape)
        return gx
    }


    function BroadcastTo(shape) {
        let result = Operation.inherit(BroadcastTo)
        result.shape = shape
        return result
    }

    BroadcastTo.prototype.__proto__ = Operation.prototype

    BroadcastTo.prototype.forward = function(x) {

    }

    BroadcastTo.prototype.backward = function(gy) {
        
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

    function sum(x, axis, keepdims) {
        axis = axis === undefined ? null : axis
        keepdims = keepdims === undefined ? null : keepdims
        return Sum(axis, keepdims)(x)
    }


    module.exports = {
        sin : sin,
        cos : cos,
        tanh : tanh,
        exp : exp,
        log : log
    }
})()