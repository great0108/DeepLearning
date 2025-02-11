const {Variable, no_grad, test_mode} = require("./core")
const {sigmoid, mean_squared_error, softmax_cross_entropy, accuracy, dropout, relu, polling, flatten, im2col, conv2d_simple} = require("./functions")
const {Layer, Linear, Conv2d} = require("./layers")
const {MLP} = require("./models")
const {SGD, MomentumSGD, AdaGrad, AdaDelta, Adam} = require("./optimizers")
const Arr = require("./Arr")
const {Spiral, Mnist} = require("./datasets")
const {DataLoader} = require("./dataloaders")

// let x1 = Arr.rand(1, 3, 7, 7)
// let col1 = im2col(x1, 5, 1, 0, true)
// console.log(col1.shape)

// let x2 = Arr.rand(10, 3, 7, 7)
// let kernel_size = [5, 5]
// let stride = [1, 1]
// let pad = [0, 0]
// let col2 = im2col(x2, kernel_size, stride, pad, true)
// console.log(col2.shape)

// let [N, C, H, W1] = [1, 5, 15, 15]
// let [OC, KH, KW] = [8, 3, 3]
// let x = new Variable(Arr.rand(N, C, H, W1))
// let W = Arr.rand(OC, C, KH, KW)
// let y = conv2d_simple(x, W, null, 1, 1)
// y.backward()
// console.log(y.shape)
// console.log(x.grad.shape)

let a = Arr([[1,2], [3,4]])
console.log(a.mean(1))  // [2,4]
console.log(a.mean(0))  // [3,4]
console.log(a.mean(1, true))  // [[2], [4]]
console.log(a.mean(-1))  // [2,4]
console.log(a.mean([0, 1]))  // [4]
console.log(a.mean())  // [2,4]