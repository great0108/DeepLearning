(function() {
    "use strict"
    const Arr = require("./Arr")

    function Dataset(train, transform, target_transform) {
        this.train = train
        this.transform = transform === undefined ? a => a : transform
        this.target_transform = target_transform === undefined ? a => a : target_transform
        this.data = null
        this.label = null
        this.prepare()
    }

    Dataset.prototype.getItem = function() {
        let a = Array.from(arguments)
        let method = a.shift()
        if(this.label === null) {
            return [this.transform(this.data[method].apply(this.data, a)), null]
        } else {
            return [this.transform(this.data[method].apply(this.data, a)), this.target_transform(this.label[method].apply(this.label, a))]
        }
    }

    Dataset.prototype.select = function(indexs) {
        return this.getItem("select", indexs)
    }

    Dataset.prototype.slice = function(start, end) {
        return this.slice("slice", start, end)
    }

    Dataset.prototype.get = function(index) {
        return this.get("get", index)
    }

    Object.defineProperty(Dataset.prototype, "length", {
        get() {return this.data.length}
    })

    Dataset.prototype.prepare = function() {
        throw new Error("NotImplemented")
    }


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


    function Spiral() {
        Dataset.apply(this, Array.from(arguments))
    }

    Spiral.prototype.__proto__ = Dataset.prototype

    Spiral.prototype.prepare = function() {
        let [x, t] = get_spiral()
        this.data = x
        this.label = t
    }

    module.exports = {
        Spiral : Spiral
    }
})()