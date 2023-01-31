(function() {
    "use strict"
    const Arr = require("./Arr")
    const F = require("./functions")
    const {Layer, Linear} = require("./layers")

    function MLP(fc_output_sizes, activation) {
        let result = Layer.inherit(MLP)
        result.activation = activation === undefined ? F.sigmoid : activation
        result.layers = []

        for(let i = 0; i < fc_output_sizes.length; i++) {
            let layer = Linear(fc_output_sizes[i])
            result.set("l"+i, layer)
            result.layers.push(layer)
        }
        return result
    }

    MLP.prototype.__proto__ = Layer.prototype

    MLP.prototype.forward = function(x) {
        for(let i = 0; i < this.layers.length-1; i++) {
            x = this.activation(this.layers[i](x))
        }
        return this.layers[this.layers.length-1](x)
    }

    module.exports = {
        MLP : MLP
    }
})()