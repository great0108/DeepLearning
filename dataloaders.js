(function() {
    "use strict"
    const Arr = require("./Arr")

    function DataLoader(dataset, batch_size, shuffle) {
        if(!(this instanceof DataLoader)) {
            return new DataLoader(dataset, batch_size, shuffle)
        }
        this.dataset = dataset
        this.batch_size = batch_size
        this.shuffle = shuffle === undefined ? true : shuffle
        this.data_size = dataset.length
        this.max_iter = Math.ceil(this.data_size / batch_size)

        this.reset()
    }

    DataLoader.prototype.reset = function() {
        if(this.shuffle) {
            this.index = Arr.range(this.dataset.length).shuffle()
        } else {
            this.index = Arr.range(this.dataset.length)
        }
    }

    DataLoader.prototype[Symbol.iterator] = function* () {
        this.reset()
        for(let i = 0; i < this.max_iter; i++) {
            let batch_index = this.index.slice(i * this.batch_size, (i+1) * this.batch_size)
            if(i === this.max_iter-1) {
                this.reset()
            }
            let batch = batch_index.map(v => this.dataset.get(v))
            yield [batch.map(v => v[0]), batch.map(v => v[1])]
        }
    }


    function SeqDataLoader(dataset, batch_size) {
        if(!(this instanceof SeqDataLoader)) {
            return new SeqDataLoader(dataset, batch_size)
        }
        DataLoader.call(this, dataset, batch_size, false)
    }

    SeqDataLoader.prototype.__proto__ = DataLoader.prototype

    SeqDataLoader.prototype[Symbol.iterator] = function* () {
        this.reset()
        for(let i = 0; i < this.max_iter; i++) {
            let jump = Math.floor(this.data_size / this.batch_size)
            let batch = Arr()
            for(let j = 0; j < this.batch_size; j++) {
                batch.push(this.dataset.get(j * jump + i))
            }
            yield [batch.map(v => v[0]), batch.map(v => v[1])]
        }
    }


    module.exports = {
        DataLoader : DataLoader,
        SeqDataLoader : SeqDataLoader
    }
})()