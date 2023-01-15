(function() {
    "use strict"
    const Arr = require("./Arr")

    function List() {
        let arr = Array.from(arguments)
        arr.__proto__ = List.prototype
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


    function as_array(x) {
        if(typeof x === "number") {
            return Arr(x)
        }
        return x
    }


    function Function() {
        let result = function() {
            return result.__call__.apply(result, arguments);
        }
        result.__proto__ = Function.prototype;
        return result;
    }

    Function.prototype.__call__ = function() {
        let inputs = Array.from(arguments)
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

    Function.prototype.forward = function() {
        throw new Error("NotImplemented")
    }

    Function.prototype.backward = function() {
        throw new Error("NotImplemented")
    }


    function Square() {
        let result = Function.call(this)
        result.__proto__ = Square.prototype
        return result
    }

    Square.prototype.__proto__ = Function.prototype

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
        let result = Function.call(this)
        result.__proto__ = Exp.prototype
        return result
    }

    Exp.prototype.__proto__ = Function.prototype

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
        let result = Function.call(this)
        result.__proto__ = Add.prototype
        return result
    }

    Add.prototype.__proto__ = Function.prototype

    Add.prototype.forward = function(x0, x1) {
        let y = x0.plus(x1)
        return y
    }

    Add.prototype.backward = function(gy) {
        return List(gy, gy)
    }


    function Mul() {
        let result = Function.call(this)
        result.__proto__ = Mul.prototype
        return result
    }

    Mul.prototype.__proto__ = Function.prototype

    Mul.prototype.forward = function(x0, x1) {
        let y = x0.mul(x1)
        return y
    }

    Mul.prototype.backward = function(gy) {
        let x0 = this.inputs[0].data
        let x1 = this.inputs[1].data
        return List(x1.mul(gy), x0.mul(gy))
    }


    function square(x) {
        return new Square()(x)
    }

    function exp(x) {
        return new Exp()(x)
    }

    function add(x0, x1) {
        return new Add()(x0, x1)
    }

    function mul(x0, x1) {
        return new Mul()(x0, x1)
    }


    function numerical_diff(f, x, eps=1e-4) {
        let x0 = new Variable(x.data.minus(eps))
        let x1 = new Variable(x.data.plus(eps))
        let y0 = f(x0)
        let y1 = f(x1)
        return y1.data.minus(y0.data).div(2 * eps)
    }


    Variable.prototype.add = function(x) {return add(this, x)}
    Variable.prototype.mul = function(x) {return mul(this, x)}

    let a = new Variable(Arr(3))
    let b = new Variable(Arr(2))
    let c = new Variable(Arr(1))

    let y = a.mul(b).add(c)
    y.backward()

    console.log(y.view)
    console.log(a.grad)
    console.log(b.grad)

})()