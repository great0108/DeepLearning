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
        throw new Error("NotImplemented")
    }

    Optimizer.prototype.add_hook = function(f) {
        this.hooks.push(f)
    }


    function SGD(lr) {
        if(!(this instanceof SGD)) {
            return new SGD(lr)
        }
        Optimizer.call(this)
        this.lr = lr === undefined ? 0.01 : lr
    }

    SGD.prototype.__proto__ = Optimizer.prototype

    SGD.prototype.update = function() {
        let params = this.target.params().filter(p => p.grad !== null)
        for(let f of this.hooks) {
            f(params)
        }

        for(let param of params) {
            param.data = param.data.minus(param.grad.data.mul(this.lr))
        }
    }


    function MomentumSGD(lr, momentum) {
        if(!(this instanceof MomentumSGD)) {
            return new MomentumSGD(lr, momentum)
        }
        Optimizer.call(this)
        this.lr = lr === undefined ? 0.01 : lr
        this.momentum = momentum === undefined ? 0.9 : momentum
        this.vs = []
    }

    MomentumSGD.prototype.__proto__ = Optimizer.prototype

    MomentumSGD.prototype.update = function() {
        let params = this.target.params().filter(p => p.grad !== null)
        for(let f of this.hooks) {
            f(params)
        }

        if(this.vs.length === 0) {
            for(let param of params) {
                this.vs.push(Arr.zeros(param.shape))
            }
        }

        for(let i = 0; i < params.length; i++) {
            let param = params[i]
            let v = this.vs[i]
            let momentum = this.momentum
            let lr = this.lr

            this.vs[i] = v.mul(momentum).minus(param.grad.data.mul(lr))
            params[i].data = param.data.plus(this.vs[i])
        }
    }


    function AdaGrad(lr, eps) {
        if(!(this instanceof AdaGrad)) {
            return new AdaGrad(lr, eps)
        }
        Optimizer.call(this)
        this.lr = lr === undefined ? 0.01 : lr
        this.eps = eps === undefined ? 1e-8 : eps
        this.hs = []
    }

    AdaGrad.prototype.__proto__ = Optimizer.prototype

    AdaGrad.prototype.update = function() {
        let params = this.target.params().filter(p => p.grad !== null)
        for(let f of this.hooks) {
            f(params)
        }

        if(this.hs.length === 0) {
            for(let param of params) {
                this.hs.push(Arr.zeros(param.shape))
            }
        }

        for(let i = 0; i < params.length; i++) {
            let param = params[i]
            let grad = param.grad.data
            let h = this.hs[i]
            let lr = this.lr
            let eps = this.eps

            this.hs[i] = h.plus(grad.pow(2))
            param.data = param.data.minus(grad.mul(lr).div(this.hs[i].deepMap(v => Math.sqrt(v) + eps)))
        }
    }


    function AdaDelta(rho, eps) {
        if(!(this instanceof AdaDelta)) {
            return new AdaDelta(rho, eps)
        }
        Optimizer.call(this)
        this.rho = rho === undefined ? 0.95 : rho
        this.eps = eps === undefined ? 1e-6 : eps
        this.msg = []
        this.msdx = []
    }

    AdaDelta.prototype.__proto__ = Optimizer.prototype

    AdaDelta.prototype.update = function() {
        let params = this.target.params().filter(p => p.grad !== null)
        for(let f of this.hooks) {
            f(params)
        }

        if(this.msg.length === 0) {
            for(let param of params) {
                this.msg.push(Arr.zeros(param.shape))
                this.msdx.push(Arr.zeros(param.shape))
            }
        }

        for(let i = 0; i < params.length; i++) {
            let msg = this.msg[i]
            let msdx = this.msdx[i]
            let rho = this.rho
            let eps = this.eps
            let param = params[i]
            let grad = param.grad.data

            this.msg[i] = msg.mul(rho).plus(grad.pow(2).mul(1 - rho))
            let dx = msdx.plus(eps).div(this.msg[i].plus(eps)).deepMap(v => Math.sqrt(v)).mul(grad)
            this.msdx[i] = msdx.mul(rho).plus(dx.pow(2).mul(1 - rho))
            param.data = param.data.minus(dx)
        }
    }


    function Adam(alpha, beta1, beta2, eps) {
        if(!(this instanceof Adam)) {
            return new Adam(alpha, beta1, beta2, eps)
        }
        Optimizer.call(this)
        this.t = 0
        this.alpha = alpha === undefined ? 0.01 : alpha
        this.beta1 = beta1 === undefined ? 0.9 : beta1
        this.beta2 = beta2 === undefined ? 0.999 : beta2
        this.eps = eps === undefined ? 1e-8 : eps
        this.ms = []
        this.vs = []
    }

    Adam.prototype.__proto__ = Optimizer.prototype

    Adam.prototype.lr = function() {
        let fix1 = 1 - Math.pow(this.beta1, this.t)
        let fix2 = 1 - Math.pow(this.beta2, this.t)
        return this.alpha * Math.sqrt(fix2) / fix1
    }

    Adam.prototype.update = function() {
        this.t += 1
        let params = this.target.params().filter(p => p.grad !== null)
        for(let f of this.hooks) {
            f(params)
        }

        if(this.ms.length === 0) {
            for(let param of params) {
                this.ms.push(Arr.zeros(param.shape))
                this.vs.push(Arr.zeros(param.shape))
            }
        }

        for(let i = 0; i < params.length; i++) {
            let m = this.ms[i]
            let v = this.vs[i]
            let beta1 = this.beta1
            let beta2 = this.beta2
            let eps = this.eps
            let param = params[i]
            let grad = param.grad.data

            this.ms[i] = m.plus(grad.minus(m).mul(1 - beta1))
            this.vs[i] = v.plus(grad.pow(2).minus(v).mul(1 - beta2))
            param.data = param.data.minus(this.ms[i].mul(this.lr()).div(this.vs[i].deepMap(v => Math.sqrt(v) + eps)))
        }
    }

    module.exports = {
        SGD : SGD,
        MomentumSGD : MomentumSGD,
        AdaGrad : AdaGrad,
        AdaDelta : AdaDelta,
        Adam : Adam
    }
})()