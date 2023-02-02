(function() {
    "use strict"
    const Arr = require("./Arr")

    function get_spiral() {
        let num_data = 100
        let num_class = 3
        let input_dim = 2
        let data_size = num_class * num_data

        let x = Arr.zeros(data_size, input_dim)
        let t = Arr.zeros(data_size)

        for(let j = 0; j < num_class; j++) {
            for(let i = 0; i < num_data; i++) {
                let rate = i / num_data
                let theta = j * 4 + 4 * rate + Math.random() * 0.2
                let ix = num_data * j + i
                x[ix] = Arr([rate * Math.sin(theta), rate * Math.cos(theta)])
                t[ix] = j
            }
        }

        let index = Arr.range(data_size).shuffle()
        let x2 = x.copy()
        let t2 = t.copy()
        for(let i = 0; i < data_size; i++) {
            x[i] = x2[index[i]]
            t[i] = t2[index[i]]
        }
        return [x, t]
    }

    module.exports = {
        get_spiral : get_spiral
    }
})()