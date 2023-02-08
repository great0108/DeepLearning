const {Variable, no_grad, test_mode} = require("./core")
const {sigmoid, mean_squared_error, softmax_cross_entropy, accuracy, dropout} = require("./functions")
const {Layer, Linear} = require("./layers")
const {MLP} = require("./models")
const {SGD, MomentumSGD, AdaGrad, AdaDelta, Adam} = require("./optimizers")
const Arr = require("./Arr")
const {Spiral, Mnist} = require("./datasets")
const {DataLoader} = require("./dataloaders")
const utils = require("./utils")
const setting = require("./setting")

let x = Arr.fill(5, 1)
console.log(x)

let y = dropout(x)
console.log(y)

test_mode(() => {
    y = dropout(x)
    console.log(y)
})