(function() {
    "use strict"
    const Arr = require("./Arr")

    function List(arr) {
        arr.__proto__ = List.prototype
        return arr
    }
    List.prototype.__proto__ = Array.prototype

    function Variable(data) {
        if(data !== null) {
            if(!(data instanceof Arr)) {
                throw new Error(data.constructor.name + " is not supported")
            }
        }

        this.data = data
        this.grad = null
        this.creator = null
    }

    Variable.prototype.set_creator = function(func) {
        this.creator = func
    }

    Variable.prototype.backward = function() {
        if(this.grad === null) {
            this.grad = Arr.fill(this.data.shape(), 1)
        }

        let funcs = [this.creator]
        while(funcs.length > 0) {
            let f = funcs.pop()
            let gys = f.outputs.map(output => output.grad)
            let gxs = f.backward.apply(f, gys)
            if(gxs instanceof List) {
                gxs = List(gxs)
            }

            for(let i = 0; i < gxs.length; i++) {
                let x = f.inputs[i]
                let gx = gxs[i]
                x.grad = gx

                if(x.creator !== null) {
                    funcs.push(x.creator)
                }
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
        if(ys instanceof List) {
            ys = List(ys)
        }
        let outputs = ys.map(y => new Variable(as_array(y)))

        outputs.forEach(output => output.set_creator(this))
        this.inputs = inputs
        this.outputs = outputs
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
        return List([gy, gy])
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

    function numerical_diff(f, x, eps=1e-4) {
        let x0 = new Variable(x.data.minus(eps))
        let x1 = new Variable(x.data.plus(eps))
        let y0 = f(x0)
        let y1 = f(x1)
        return y1.data.minus(y0.data).div(2 * eps)
    }

    let x = new Variable(Arr(2))
    let y = new Variable(Arr(3))

    let z = add(square(x), square(y))
    z.backward()
    console.log(z.data)
    console.log(x.grad)
    console.log(y.grad)


})()