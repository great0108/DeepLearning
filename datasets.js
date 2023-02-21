(function() {
    "use strict"
    const Arr = require("./Arr")
    const utils = require("./utils")

    function Dataset(train, transform, target_transform, length) {
        this.train = train === undefined ? true : train
        this.transform = !transform ? a => a : transform
        this.target_transform = !target_transform ? a => a : target_transform
        this.data = null
        this.label = null
        this.prepare(length)
        this.data = this.transform(this.data)
        this.label = this.target_transform(this.label)
    }

    Dataset.prototype.slice = function(start, end) {
        if(this.label === null) {
            return [this.data.slice(start, end), null]
        } else {
            return [this.data.slice(start, end), this.label.slice(start, end)]
        }
    }

    Dataset.prototype.get = function(index) {
        if(this.label === null) {
            return [this.data[index], null]
        } else {
            return [this.data[index], this.label[index]]
        }
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
        if(!(this instanceof Spiral)) {
            return new Spiral()
        }
        Dataset.apply(this, Array.from(arguments))
    }

    Spiral.prototype.__proto__ = Dataset.prototype

    Spiral.prototype.prepare = function() {
        let [x, t] = get_spiral()
        this.data = x
        this.label = t
    }


    function Mnist(train, transform, target_transform, length) {
        if(!(this instanceof Mnist)) {
            return new Mnist(train, transform, target_transform, length)
        }
        transform = !transform ? a => a.deepMap(v => Number(v) / 255) : transform
        target_transform = !target_transform ? a => a.deepMap(v => Number(v)).argmax(1) : target_transform
        Dataset.call(this, train, transform, target_transform, length)
    }

    Mnist.prototype.__proto__ = Dataset.prototype

    Mnist.prototype.prepare = function(length) {
        let data = null
        let label = null
        if(this.train) {
            data = utils.read_csv("./mnist_csv/csv_image.csv")
            label = utils.read_csv("./mnist_csv/csv_label.csv")
        } else {
            data = utils.read_csv("./mnist_csv/csv_image_test.csv")
            label = utils.read_csv("./mnist_csv/csv_label_test.csv")
        }

        if(length !== undefined) {
            let index = Arr.range(data.length).shuffle().slice(0, data.length * length)
            data = index.map(v => data[v])
            label = index.map(v => label[v])
        }

        this.data = data
        this.label = label
    }

    module.exports = {
        Spiral : Spiral,
        Mnist : Mnist
    }
})()