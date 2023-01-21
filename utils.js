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
        if(!(ndim === 0 || array_axis !== null || keepdims)) {
            let actual_axis = array_axis.map(a => a >= 0 ? a : a + ndim)
            for(a of actual_axis.sort()) {
                shape.splice(a, 0, 1)
            }
        }

        gy = gy.reshape(shape)
        return gy
    }

    module.exports = {
        reshape_sum_backward : reshape_sum_backward
    }
})