(function() {
    "use strict"
    const Arr = require("./Arr")

    function reshape_sum_backward(gy, x_shape, axis, keepdims) {
        let ndim = x_shape.length
        let array_axis = axis
        if(axis !== null && !Array.isArray(axis)) {
            array_axis = [array_axis]
        }

        let shape = gy.shape
        if(!(ndim === 0 || array_axis === null || keepdims)) {
            let actual_axis = array_axis.map(a => a >= 0 ? a : a + ndim)
            for(let a of actual_axis.sort((a, b) => a-b)) {
                shape.splice(a, 0, 1)
            }
        }

        gy = gy.reshape(shape)
        return gy
    }

    function sum_to(x, shape) {
        let ndim = shape.length
        let lead = x.ndim - ndim
        let lead_axis = Arr.range(lead)

        let axis = []
        for(let i = 0; i < ndim; i++) {
            if(shape[i] === 1) {
                axis.push(i + lead)
            }
        }

        let y = x.sum(lead_axis.concat(axis), true)
        if(lead > 0) {
            y = y.squeeze(lead_axis)
        }
        return y
    }

    module.exports = {
        reshape_sum_backward : reshape_sum_backward,
        sum_to : sum_to
    }
})()