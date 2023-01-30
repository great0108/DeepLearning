(function() {
    "use strict"
    const Arr = require("./Arr")
    const {Operation, Variable, List, sum_to} = require("./core")

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


    function MatMul() {
        return Operation.inherit(MatMul)
    }

    MatMul.prototype.__proto__ = Operation.prototype

    MatMul.prototype.forward = function(x, W) {
        let y = x.matmul(W)
        return y
    }

    MatMul.prototype.backward = function(gy) {
        let x = this.inputs[0]
        let W = this.inputs[1]
        let gx = matmul(gy, W.T)
        let gW = matmul(x.T, gy)
        return List(gx, gW)
    }


    function Linear() {
        return Operation.inherit(Linear)
    }

    Linear.prototype.__proto__ = Operation.prototype

    Linear.prototype.forward = function(x, W, b) {
        let y = x.matmul(W)
        if(b != null) {
            y = y.plus(b)
        }
        return y
    }

    Linear.prototype.backward = function(gy) {
        let [x, W, b] = this.inputs
        let gb = b.data != null ? sum_to(gy, b.shape) : null
        let gx = matmul(gy, W.T)
        let gW = matmul(x.T, gy)
        return List(gx, gW, gb)
    }


    function Sigmoid() {
        return Operation.inherit(Sigmoid)
    }

    Sigmoid.prototype.__proto__ = Operation.prototype

    Sigmoid.prototype.forward = function(x) {
        let y = x.mul(0.5).deepMap(v => Math.tanh(v)).mul(0.5).plus(0.5)
        return y
    }

    Sigmoid.prototype.backward = function(gy) {
        let y = this.outputs[0]
        let gx = gy.mul(y).mul(y.rminus(1))
        return gx
    }


    function MeanSquaredError() {
        return Operation.inherit(MeanSquaredError)
    }

    MeanSquaredError.prototype.__proto__ = Operation.prototype

    MeanSquaredError.prototype.forward = function(x0, x1) {
        let diff = x0.minus(x1)
        let y = diff.pow(2).sum() / diff.length
        return y
    }

    MeanSquaredError.prototype.backward = function(gy) {
        let x0 = this.inputs[0]
        let x1 = this.inputs[1]
        let diff = x0.minus(x1)
        let gx0 = gy.mul(diff).mul(2 / diff.length)
        let gx1 = gx0.mul(-1)
        return List(gx0, gx1)
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

    function matmul(x, W) {
        return MatMul()(x, W)
    }

    function linear(x, W, b) {
        b = b === undefined ? null : b
        return Linear()(x, W, b)
    }

    function sigmoid(x) {
        return Sigmoid()(x)
    }

    function mean_squared_error(x0, x1) {
        return MeanSquaredError()(x0, x1)
    }


    module.exports = {
        sin : sin,
        cos : cos,
        tanh : tanh,
        exp : exp,
        log : log,
        matmul : matmul,
        linear : linear,
        sigmoid : sigmoid,
        mean_squared_error : mean_squared_error
    }
})()