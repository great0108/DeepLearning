const {Variable, no_grad, test_mode} = require("./core")
const {sigmoid, mean_squared_error, softmax_cross_entropy, accuracy, dropout, relu, polling, flatten} = require("./functions")
const {Layer, Linear, Conv2d} = require("./layers")
const {MLP} = require("./models")
const {SGD, MomentumSGD, AdaGrad, AdaDelta, Adam} = require("./optimizers")
const Arr = require("./Arr")
const {Spiral, Mnist} = require("./datasets")
const {DataLoader} = require("./dataloaders")

function get_conv_outsize(input_size, kernel_size, stride, pad) {
    return Math.floor((input_size + pad * 2 - kernel_size) / stride) + 1
}


let [H, W] = [4, 4]
let [KH, KW] = [3, 3]
let [SH, SW] = [1, 1]
let [PH, PW] = [1, 1]

let OH = get_conv_outsize(H, KH, SH, PH)
let OW = get_conv_outsize(W, KW, SW, PW)
console.log(OH, OW)