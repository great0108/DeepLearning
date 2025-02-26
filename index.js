(function() {
    "use strict"
    const Arr = require("./Arr")
    const core = require("./core")
    const functions = require("./functions")
    const layers = require("./layers")
    const optimizers = require("./optimizers")
    const datasets = require("./datasets")
    const dataloaders = require("./dataloaders")
    const utils = require("./utils")

    module.exports = {
        Arr : Arr,
        core : core,
        functions : functions,
        layers : layers,
        optimizers : optimizers,
        datasets : datasets,
        dataloaders : dataloaders,
        utils : utils
    }
})()