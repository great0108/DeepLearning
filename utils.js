(function() {
    "use strict"
    const Arr = require("./Arr")
    const setting = require("./setting")

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

    function read_csv(path) {
        let csv = null
        if(setting.nodeJS) {
            const fs = require("fs")
            csv = fs.readFileSync(path, "utf-8")
        } else {
            csv = FileStream.read(setting.module_path + "/" + path)
        }
        const rows = csv.split("\n")
        const result = rows.map(v => v.split(","))
        return Arr(result)
    }

    function read_json(path) {
        let json = null
        if(setting.nodeJS) {
            const fs = require("fs")
            json = fs.readFileSync(path, "utf-8")
        } else {
            json = FileStream.read(setting.module_path + "/" + path)
        }
        json = JSON.parse(json)
        Object.keys(json).forEach(key => json[key] = Arr(json[key]))
        return json
    }

    function write_json(path, json) {
        json = JSON.stringify(json)
        if(setting.nodeJS) {
            const fs = require("fs")
            fs.writeFileSync(path, json, "utf-8")
        } else {
            FileStream.write(setting.module_path + "/" +  path, json)
        }
    }

    module.exports = {
        reshape_sum_backward : reshape_sum_backward,
        sum_to : sum_to,
        logsumexp : logsumexp,
        max_backward_shape : max_backward_shape,
        read_csv : read_csv,
        read_json : read_json,
        write_json : write_json
    }
})()