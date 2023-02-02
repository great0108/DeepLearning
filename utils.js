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

    function logsumexp(x, axis) {
        axis = axis === undefined ? 1 : axis
        let m = x.max(axis, true)
        let y = x.minus(m)
        y = y.deepMap(v => Math.exp(v))
        let s = y.sum(axis, true)
        s = s.deepMap(v => Math.log(v))
        m = m.plus(s)
        return m
    }

    function max_backward_shape(x, axis) {
        if(axis === null) {
            axis = Arr.range(x.ndim)
        } else if(!isNaN(axis)) {
            axis = [axis]
        }

        let shape = []
        let x_shape = x.shape
        for(let i = 0; i < x_shape.length; i++) {
            if(axis.includes(i)) {
                shape.push(1)
            } else {
                shape.push(x_shape[i])
            }
        }
        return shape
    }

    module.exports = {
        reshape_sum_backward : reshape_sum_backward,
        sum_to : sum_to,
        logsumexp : logsumexp,
        max_backward_shape : max_backward_shape
    }
})()