(function() {
    "use strict"

    function DataLoader(dataset, batch_size, shuffle) {
        this.dataset = dataset
        this.batch_size = batch_size
        this.shuffle = shuffle === undefined ? true : shuffle
        this.data_size = dataset.length
        this.max_iter = Math.ceil(this.data_size / this.batch_size)
        this.reset()
    }

    DataLoader.prototype.reset = 1

    module.exports = {
        
    }
})()