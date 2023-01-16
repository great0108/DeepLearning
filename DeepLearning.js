(function() {
    "use strict"
    const Arr = require("./Arr")

    function List() {
        let arr = Array.from(arguments)
        Object.setPrototypeOf(arr, List.prototype)
        return arr
    }
    List.prototype.__proto__ = Array.prototype

    const Config = {
        "enable_backprop" : true
    }

    function using_config(name, value, callback) {
        let old_value = Config[name]
        Config[name] = value
        try {
            callback()
        } finally {
            Config[name] = old_value
        }
    }

    function no_grad(callback) {
        using_config("enable_backprop", false, callback)
    }


    function Variable(data, name) {
        if(data !== null) {
            if(!(data instanceof Arr)) {
                throw new Error(data.constructor.name + " is not supported")
            }
        }

        this.data = data
        this.name = name === undefined ? null : name
        this.grad = null
        this.creator = null
        this.generation = 0
    }

    Object.defineProperties(Variable.prototype, {
        shape : {
            get() {return this.data.shape()}
        },
        ndim : {
            get() {return this.data.ndim()}
        },
        size : {
            get() {return this.data.size()}
        },
        length : {
            get() {return this.data.length}
        },
        view : {
            get() {return this.data.view()}
        }
    })

    Variable.prototype.set_creator = function(func) {
        this.creator = func
        this.generation = func.generation + 1
    }

    Variable.prototype.cleargrad = function() {
        this.grad = null
    }

    Variable.prototype.backward = function(retain_grad) {
        if(this.grad === null) {
            this.grad = Arr.fill(this.data.shape(), 1)
        }

        let funcs = []
        let seen_set = new Set()

        function add_func(f) {
            if(!(seen_set.has(f))) {
                funcs.push(f)
                seen_set.add(f)
                funcs.sort((a, b) => a.generation - b.generation)
            }
        }

        add_func(this.creator)

        while(funcs.length > 0) {
            let f = funcs.pop()
            let gys = f.outputs.map(output => as_array(output.grad))
            let gxs = f.backward.apply(f, gys)
            if(!(gxs instanceof List)) {
                gxs = List(gxs)
            }

            for(let i = 0; i < gxs.length; i++) {
                let x = f.inputs[i]
                let gx = gxs[i]
                if(x.grad === null) {
                    x.grad = gx
                } else {
                    x.grad = x.grad.plus(gx)
                }

                if(x.creator !== null) {
                    add_func(x.creator)
                }
            }

            if(!retain_grad) {
                f.outputs.forEach(y => y.grad = null)
            }
        }
    }


    function as_variable(obj) {
        if(obj instanceof Variable) {
            return obj
        }
        return new Variable(obj)
    }

    function as_array(x) {
        if(typeof x === "number") {
            return Arr(x)
        }
        return x
    }


    function Operation() {
        let result = function() {
            return result.__call__.apply(result, arguments);
        }
        Object.setPrototypeOf(result, Operation.prototype)
        return result;
    }

    Operation.prototype.__call__ = function() {
        let inputs = Array.from(arguments).map(x => as_variable(x))
        let xs = inputs.map(x => x.data)
        let ys = this.forward.apply(this, xs)
        if(!(ys instanceof List)) {
            ys = List(ys)
        }
        let outputs = ys.map(y => new Variable(as_array(y)))

        if(Config.enable_backprop) {
            this.generation = Math.max.apply(null, inputs.map(x => x.generation))
            outputs.forEach(output => output.set_creator(this))
            this.inputs = inputs
            this.outputs = outputs
        }
        return outputs.length > 1 ? outputs : outputs[0]
    }

    Operation.prototype.forward = function() {
        throw new Error("NotImplemented")
    }

    Operation.prototype.backward = function() {
        throw new Error("NotImplemented")
    }


    function Square() {
        let result = Operation.call(this)
        Object.setPrototypeOf(result, Square.prototype)
        return result
    }

    Square.prototype.__proto__ = Operation.prototype

    Square.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.pow(v, 2))
        return y
    }

    Square.prototype.backward = function(gy) {
        let x = this.inputs[0].data
        let gx = x.mul(2).mul(gy)
        return gx
    }


    function Exp() {
        let result = Operation.call(this)
        Object.setPrototypeOf(result, Exp.prototype)
        return result
    }

    Exp.prototype.__proto__ = Operation.prototype

    Exp.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.exp(v))
        return y
    }

    Exp.prototype.backward = function(gy) {
        let x = this.inputs[0].data
        let gx = x.deepMap(v => Math.exp(v)).mul(gy)
        return gx
    }

    
    function Add() {
        let result = Operation.call(this)
        Object.setPrototypeOf(result, Add.prototype)
        return result
    }

    Add.prototype.__proto__ = Operation.prototype

    Add.prototype.forward = function(x0, x1) {
        let y = x0.plus(x1)
        return y
    }

    Add.prototype.backward = function(gy) {
        return List(gy, gy)
    }


    function Mul() {
        let result = Operation.call(this)
        Object.setPrototypeOf(result, Mul.prototype)
        return result
    }

    Mul.prototype.__proto__ = Operation.prototype

    Mul.prototype.forward = function(x0, x1) {
        let y = x0.mul(x1)
        return y
    }

    Mul.prototype.backward = function(gy) {
        let x0 = this.inputs[0].data
        let x1 = this.inputs[1].data
        return List(x1.mul(gy), x0.mul(gy))
    }


    function Neg() {
        let result = Operation.call(this)
        Object.setPrototypeOf(result, Neg.prototype)
        return result
    }

    Neg.prototype.__proto__ = Operation.prototype

    Neg.prototype.forward = function(x) {
        return x.mul(-1)
    }

    Neg.prototype.backward = function(gy) {
        return gy.mul(-1)
    }

    
    function Sub() {
        let result = Operation.call(this)
        Object.setPrototypeOf(result, Sub.prototype)
        return result
    }

    Sub.prototype.__proto__ = Operation.prototype

    Sub.prototype.forward = function(x0, x1) {
        let y = x0.minus(x1)
        return y
    }

    Sub.prototype.backward = function(gy) {
        return List(gy, gy.mul(-1))
    }


    function Div() {
        let result = Operation.call(this)
        Object.setPrototypeOf(result, Div.prototype)
        return result
    }

    Div.prototype.__proto__ = Operation.prototype

    Div.prototype.forward = function(x0, x1) {
        let y = x0.div(x1)
        return y
    }

    Div.prototype.backward = function(gy) {
        let x0 = this.inputs[0].data
        let x1 = this.inputs[1].data
        let gx0 = gy.div(x1)
        let gx1 = gy.mul(x0.mul(-1).div(x1.deepMap(v => Math.pow(v, 2))))
        return List(gx0, gx1)
    }


    function Pow(c) { // c가 상수라고 가정했음
        let result = Operation.call(this)
        Object.setPrototypeOf(result, Pow.prototype)
        result.c = c
        return result
    }

    Pow.prototype.__proto__ = Operation.prototype

    Pow.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.pow(v, this.c))
        return y
    }

    Pow.prototype.backward = function(gy) {
        let x = this.inputs[0].data
        let c = this.c

        let gx = x.deepMap(v => Math.pow(v, c-1)).mul(c).mul(gy)
        return gx
    }


    function square(x) {
        return new Square()(x)
    }

    function exp(x) {
        return new Exp()(x)
    }

    function add(x0, x1) {
        x1 = as_array(x1)
        return new Add()(x0, x1)
    }

    function mul(x0, x1) {
        x1 = as_array(x1)
        return new Mul()(x0, x1)
    }

    function neg(x) {
        return new Neg()(x)
    }

    function sub(x0, x1) {
        x1 = as_array(x1)
        return new Sub()(x0, x1)
    }

    function rsub(x0, x1) {
        x1 = as_array(x1)
        return sub(x1, x0)
    }

    function div(x0, x1) {
        x1 = as_array(x1)
        return new Div()(x0, x1)
    }

    function rdiv(x0, x1) {
        x1 = as_array(x1)
        return div(x1, x0)
    }

    function pow(x, c) {
        return new Pow(c)(x)
    }


    function numerical_diff(f, x, eps) {
        eps = eps === undefined ? 1e-4 : eps
        let x0 = new Variable(x.data.minus(eps))
        let x1 = new Variable(x.data.plus(eps))
        let y0 = f(x0)
        let y1 = f(x1)
        return y1.data.minus(y0.data).div(2 * eps)
    }

    Variable.prototype.add = function(x) {return add(this, x)}
    Variable.prototype.mul = function(x) {return mul(this, x)}
    Variable.prototype.neg = function() {return neg(this)}
    Variable.prototype.sub = function(x) {return sub(this, x)}
    Variable.prototype.rsub = function(x) {return rsub(this, x)}
    Variable.prototype.div = function(x) {return div(this, x)}
    Variable.prototype.rdiv = function(x) {return rdiv(this, x)}
    Variable.prototype.pow = function(c) {return pow(this, c)}

    
    let x = new Variable(Arr(2))
    let y = x.neg()
    console.log(y.view)

    let y1 = x.rsub(2)
    let y2 = x.sub(1)
    console.log(y1.view)
    console.log(y2.view)

    y = x.rdiv(3)
    console.log(y.view)

    y = x.pow(3)
    y.backward()
    console.log(y.view)
    console.log(x.grad)
    
})()