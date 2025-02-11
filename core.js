(function() {
    "use strict"
    const Arr = require("./Arr")
    const Callable = require("./Callable")
    const utils = require("./utils")

    function List() {
        let arr = Array.from(arguments)
        Object.setPrototypeOf(arr, List.prototype)
        return arr
    }
    List.prototype.__proto__ = Array.prototype


    const Config = {
        "enable_backprop" : true,
        "train" : true
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

    function test_mode(callback) {
        using_config("train", false, callback)
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
            get() {return this.data.shape}
        },
        ndim : {
            get() {return this.data.ndim}
        },
        size : {
            get() {return this.data.size}
        },
        length : {
            get() {return this.data.length}
        },
        view : {
            get() {return this.data.view}
        },
        T : {
            get() {return transpose(this)}
        }
    })

    Variable.prototype.set_creator = function(func) {
        this.creator = func
        this.generation = func.generation + 1
    }

    Variable.prototype.cleargrad = function() {
        this.grad = null
    }

    Variable.prototype.backward = function(create_graph, retain_grad) {
        create_graph = create_graph === undefined ? false : create_graph
        if(this.grad === null) {
            this.grad = new Variable(Arr.fill(this.data.shape, 1))
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
            let gys = f.outputs.map(output => output.grad)

            using_config("enable_backprop", create_graph, () => {
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
            })

            if(!retain_grad) {
                f.outputs.forEach(y => y.grad = null)
            }
        }
    }

    Variable.prototype.copy = function(data) {
        let result = new Variable(data, this.name)
        result.grad = this.grad
        result.creator = this.creator
        result.generation = this.generation
        return result
    }

    Variable.prototype.deepMap = function(fn) {
        let data = this.data.deepMap(fn)
        return this.copy(data)
    }

    Variable.prototype.flat = function(n) {
        let data = this.data.flat(n)
        return this.copy(data)
    }

    Variable.prototype.reshape = function(shape) {
        shape = arguments.length === 1 
        ? (Array.isArray(shape) ? shape : Array.of(shape)) 
        : Array.from(arguments)
        return reshape(this, shape)
    }

    Variable.prototype.transpose = function(axes) {
        if(axes !== undefined) {
            axes = arguments.length === 1
            ? (Array.isArray(axes) ? axes : Array.of(axes))
            : Array.from(arguments)
        }
        return transpose(this, axes)
    }


    function Parameter() {
        let args = Array.from(arguments)
        Variable.apply(this, args)
    }

    Parameter.prototype.__proto__ = Variable.prototype

    
    function Operation() {
        return Callable.inherit(Operation)
    }

    Operation.inherit = Callable.inherit

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

    
    function Add() {
        return Operation.inherit(Add)
    }

    Add.prototype.__proto__ = Operation.prototype

    Add.prototype.forward = function(x0, x1) {
        this.x0_shape = x0.shape
        this.x1_shape = x1.shape
        let y = x0.plus(x1)
        return y
    }

    Add.prototype.backward = function(gy) {
        let gx0 = gy, gx1 = gy
        if(!this.x0_shape.same(this.x1_shape)) {
            gx0 = sum_to(gx0, this.x0_shape)
            gx1 = sum_to(gx1, this.x1_shape)
        }
        return List(gx0, gx1)
    }


    function Mul() {
        return Operation.inherit(Mul)
    }

    Mul.prototype.__proto__ = Operation.prototype

    Mul.prototype.forward = function(x0, x1) {
        let y = x0.mul(x1)
        return y
    }

    Mul.prototype.backward = function(gy) {
        let [x0, x1] = this.inputs
        let gx0 = x1.mul(gy)
        let gx1 = x0.mul(gy)
        if(!x0.shape.same(x1.shape)) {
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        }
        return List(gx0, gx1)
    }


    function Neg() {
        return Operation.inherit(Neg)
    }

    Neg.prototype.__proto__ = Operation.prototype

    Neg.prototype.forward = function(x) {
        return x.mul(-1)
    }

    Neg.prototype.backward = function(gy) {
        return gy.mul(-1)
    }

    
    function Sub() {
        return Operation.inherit(Sub)
    }

    Sub.prototype.__proto__ = Operation.prototype

    Sub.prototype.forward = function(x0, x1) {
        this.x0_shape = x0.shape
        this.x1_shape = x1.shape
        let y = x0.minus(x1)
        return y
    }

    Sub.prototype.backward = function(gy) {
        let gx0 = gy
        let gx1 = gy.mul(-1)
        if(!this.x0_shape.same(this.x1_shape)) {
            gx0 = sum_to(gx0, this.x0_shape)
            gx1 = sum_to(gx1, this.x1_shape)
        }
        return List(gx0, gx1)
    }


    function Div() {
        return Operation.inherit(Div)
    }

    Div.prototype.__proto__ = Operation.prototype

    Div.prototype.forward = function(x0, x1) {
        let y = x0.div(x1)
        return y
    }

    Div.prototype.backward = function(gy) {
        let x0 = this.inputs[0]
        let x1 = this.inputs[1]
        let gx0 = gy.div(x1)
        let gx1 = gy.mul(x0.mul(-1).div(x1.pow(2)))
        if(!x0.shape.same(x1.shape)) {
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        }
        return List(gx0, gx1)
    }


    function Pow(c) { // c가 상수라고 가정했음
        let result = Operation.inherit(Pow)
        result.c = c
        return result
    }

    Pow.prototype.__proto__ = Operation.prototype

    Pow.prototype.forward = function(x) {
        let y = x.pow(this.c)
        return y
    }

    Pow.prototype.backward = function(gy) {
        let x = this.inputs[0]
        let c = this.c

        let gx = x.pow(c-1).mul(c).mul(gy)
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
        let y = x.transpose(this.axes)
        return y
    }

    Transpose.prototype.backward = function(gy) {
        if(this.axes === undefined) {
            return transpose(gy)
        }

        let axes_len = this.axes.length
        let inv_axes = []
        for(let i = 0; i < axes_len; i++) {
            inv_axes.push(this.axes.indexOf(i))
        }
        return transpose(gy, inv_axes)
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
        gy = utils.reshape_sum_backward(gy, this.x_shape, this.axis, this.keepdims)
        let gx = broadcast_to(gy, this.x_shape)
        return gx
    }


    function SumTo(shape) {
        let result = Operation.inherit(SumTo)
        result.shape = shape
        return result
    }

    SumTo.prototype.__proto__ = Operation.prototype

    SumTo.prototype.forward = function(x) {
        this.x_shape = x.shape
        let y = utils.sum_to(x, this.shape)
        return y
    }

    SumTo.prototype.backward = function(gy) {
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
        this.x_shape = x.shape
        let y = x.broadcast(this.shape)
        return y
    }

    BroadcastTo.prototype.backward = function(gy) {
        let gx = sum_to(gy, this.x_shape)
        return gx
    }


    function Max(axis, keepdims) {
        let result = Operation.inherit(Max)
        result.axis = axis === undefined ? null : axis
        result.keepdims = keepdims === undefined ? false : keepdims
        return result
    }

    Max.prototype.__proto__ = Operation.prototype

    Max.prototype.forward = function(x) {
        let y = x.max(this.axis, this.keepdims)
        return y
    }

    Max.prototype.backward = function(gy) {
        let x = this.inputs[0]
        let y = this.outputs[0]

        let shape = utils.max_backward_shape(x, this.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        let cond = x.data.cal(y.data, (a, b) => Number(a === b))
        gy = broadcast_to(gy, cond.shape)
        return gy.mul(cond)
    }


    function Min(axis, keepdims) {
        let result = Max(axis, keepdims)
        Object.setPrototypeOf(result, Min.prootype)
        return result
    }

    Min.prototype.__proto__ = Max.prototype

    Min.prototype.forward = function(x) {
        let y = x.min(this.axis, this.keepdims)
        return y
    }


    function GetItem(slices) {
        let result = Operation.inherit(GetItem)
        result.slices = slices
        return result
    }

    GetItem.prototype.__proto__ = Operation.prototype

    GetItem.prototype.forward = function(x) {
        let y = x.slice(this.slices[0], this.slices[1])
        return y
    }

    GetItem.prototype.backward = function(gy) {
        let x = this.inputs[0]
        let f = GetItemGrad(this.slices, x.shape)
        return f(gy)
    }


    function GetItemGrad(slices, in_shape) {
        let result = Operation.inherit(GetItemGrad)
        result.slices = slices
        result.in_shape = in_shape
    }

    GetItemGrad.prototype.__proto__ = Operation.prototype

    GetItemGrad.prototype.forward = function(gy) {
        let gx = Arr.zeros(this.in_shape)
        gx.overlap(gy, this.slices[0], this.slices[1])
        return gx
    }

    GetItemGrad.prototype.backward = function(ggx) {
        return get_item(ggx, this.slices)
    }


    function add(x0, x1) {
        x1 = as_array(x1)
        return Add()(x0, x1)
    }

    function mul(x0, x1) {
        x1 = as_array(x1)
        return Mul()(x0, x1)
    }

    function neg(x) {
        return Neg()(x)
    }

    function sub(x0, x1) {
        x1 = as_array(x1)
        return Sub()(x0, x1)
    }

    function rsub(x0, x1) {
        x1 = as_array(x1)
        return sub(x1, x0)
    }

    function div(x0, x1) {
        x1 = as_array(x1)
        return Div()(x0, x1)
    }

    function rdiv(x0, x1) {
        x1 = as_array(x1)
        return div(x1, x0)
    }

    function pow(x, c) {
        return Pow(c)(x)
    }

    function reshape(x, shape) {
        if(x.shape.same(shape)) {
            return as_variable(x)
        }
        return Reshape(shape)(x)
    }

    function transpose(x, axes) {
        return Transpose(axes)(x)
    }

    function sum(x, axis, keepdims) {
        axis = axis === undefined ? null : axis
        keepdims = keepdims === undefined ? false : keepdims
        return Sum(axis, keepdims)(x)
    }

    function sum_to(x, shape) {
        if(x.shape.same(shape)) {
            return as_variable(x)
        }
        return SumTo(shape)(x)
    }

    function broadcast_to(x, shape) {
        if(x.shape.same(shape)) {
            return as_variable(x)
        }
        return BroadcastTo(shape)(x)
    }

    function max(x, axis, keepdims) {
        return Max(axis, keepdims)(x)
    }

    function min(x, axis, keepdims) {
        return Min(axis, keepdims)(x)
    }

    function slice(x, slices) {
        return GetItem(slices)(x)
    }

    function get(x, index) {
        let slices = null
        if(Array.isArray(index)) {
            slices = [index, index.map(v => v+1)]
        } else {
            slices = [index, index+1]
        }
        return GetItem(slices)(x)
    }


    Variable.prototype.plus = function(x) {return add(this, x)}
    Variable.prototype.mul = function(x) {return mul(this, x)}
    Variable.prototype.neg = function() {return neg(this)}
    Variable.prototype.minus = function(x) {return sub(this, x)}
    Variable.prototype.rminus = function(x) {return rsub(this, x)}
    Variable.prototype.div = function(x) {return div(this, x)}
    Variable.prototype.rdiv = function(x) {return rdiv(this, x)}
    Variable.prototype.pow = function(c) {return pow(this, c)}
    Variable.prototype.sum = function(axis, keepdims) {return sum(this, axis, keepdims)}
    Variable.prototype.max = function(axis, keepdims) {return max(this, axis, keepdims)}
    Variable.prototype.min = function(axis, keepdims) {return min(this, axis, keepdims)}
    Variable.prototype.slice = function(start, end) {return slice(this, [start, end])}
    Variable.prototype.get = function(index) {return get(this, index)}


    module.exports = {
        Variable : Variable,
        Parameter : Parameter,
        Operation : Operation,
        List : List,
        Config : Config,
        as_array : as_array,
        as_variable : as_variable,
        no_grad : no_grad,
        test_mode : test_mode,
        sum_to : sum_to,
        broadcast_to : broadcast_to
    }
})()