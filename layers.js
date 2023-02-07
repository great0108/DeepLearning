(function() {
    "use strict"
    const Arr = require("./Arr")
    const {Parameter, List} = require("./core")
    const F = require("./functions")
    const Callable = require("./Callable")
    const utils = require("./utils")

    function Layer() {
        let result = Callable.inherit(Layer)
        result._params = new Set()
        return result
    }
    
    Layer.inherit = Callable.inherit

    Layer.prototype.set = function(name, value) {
        if(value instanceof Parameter || value instanceof Layer) {
            this._params.add(name)
        }
        this[name] = value
    }

    Layer.prototype.__call__ = function() {
        let inputs = Array.from(arguments)
        let outputs = this.forward.apply(this, inputs)
        if(!(outputs instanceof List)) {
            outputs = List(outputs)
        }

        this.inputs = inputs
        this.outputs = outputs
        return outputs.length > 1 ? outputs : outputs[0]
    }

    Layer.prototype.forward = function() {
        throw new Error("NotImplemented")
    }

    Layer.prototype.params = function() {
        let names = Array.from(this._params)
        let result = []
        for(let name of names) {
            if(this[name] instanceof Layer) {
                result = result.concat(this[name].params())
            } else {
                result.push(this[name])
            }
        }
        return result
    }

    Layer.prototype.cleargrads = function() {
        for(let param of this.params()) {
            param.cleargrad()
        }
    }

    Layer.prototype._flatten_params = function(param_dict, parent_key) {
        parent_key = parent_key === undefined ? "" : parent_key
        for(let name of this._params) {
            let obj = this[name]
            let key = parent_key ? parent_key + "/" + name : name

            if(obj instanceof Layer) {
                obj._flatten_params(param_dict, key)
            } else {
                param_dict[key] = obj
            }
        }
    }

    Layer.prototype.save_weights = function(path) {
        let param_dict = {}
        this._flatten_params(param_dict)
        let array_dict = {}
        for(let key of Object.keys(param_dict)) {
            array_dict[key] = param_dict[key] === null ? null : param_dict[key].data
        }

        utils.write_json(path, array_dict)
    }

    Layer.prototype.load_weights = function(path) {
        let array_dict = utils.read_json(path)
        let param_dict = {}
        this._flatten_params(param_dict)

        for(let key of Object.keys(param_dict)) {
            param_dict[key].data = array_dict[key]
        }
    }


    function Linear(out_size, nobias, in_size) {
        let result = Layer.inherit(Linear)
        result.in_size = in_size
        result.out_size = out_size
        result.set("W", new Parameter(null, "W"))

        if(result.in_size !== undefined) {
            result._init_W()
        }

        if(nobias) {
            result.b = null
        } else {
            result.set("b", new Parameter(Arr.zeros(result.out_size), "b"))
        }
        return result
    }

    Linear.prototype.__proto__ = Layer.prototype

    Linear.prototype._init_W = function() {
        let I = this.in_size
        let O = this.out_size
        this.W.data = Arr.rand(I, O).mul(Math.sqrt(1/I))
    }

    Linear.prototype.forward = function(x) {
        if(this.W.data === null) {
            this.in_size = x.shape[1]
            this._init_W()
        }
        let y = F.linear(x, this.W, this.b)
        return y
    }

    module.exports = {
        Layer : Layer,
        Linear : Linear
    }
})()