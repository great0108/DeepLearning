(function() {
    "use strict"
    const Arr = require("./Arr")

    function Optimizer() {
        this.target = null
        this.hooks = []
    }

    Optimizer.prototype.setup = function(target) {
        this.target = target
        return this
    }

    Optimizer.prototype.update = function() {
        let params = this.target.params().filter(p => p.grad !== null)
        for(let f of this.hooks) {
            f(params)
        }

        for(let param of params) {
            this.update_one(param)
        }
    }

    Optimizer.prototype.update_one = function(param) {
        throw new Error("NotImplemented")
    }

    Optimizer.prototype.add_hook = function(f) {
        this.hooks.push(f)
    }


    function SGD(learning_rate) {
        Optimizer.call(this)
        this.lr = learning_rate === undefined ? 0.01 : learning_rate
    }

    SGD.prototype.__proto__ = Optimizer.prototype

    SGD.prototype.update_one = function(param) {
        param.data = param.data.minus(param.grad.data.mul(this.lr))
    }

    module.exports = {
        SGD : SGD
    }
})()