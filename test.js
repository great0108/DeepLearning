const {Variable} = require("./core")
const {sigmoid, mean_squared_error} = require("./functions")
const {Layer, Linear} = require("./layers")
const {MLP} = require("./models")
const {SGD, MomentumSGD, AdaGrad, AdaDelta, Adam} = require("./optimizers")
const Arr = require("./Arr")

// let x = Arr.rand(100, 1)
// let y = x.deepMap(v => Math.sin(2 * Math.PI * v) + Math.random())

// let lr = 0.2
// let iters = 1000

// let model = MLP([20, 10, 1])
// let optimizer = AdaGrad(lr).setup(model)

// for(let i = 0; i < iters; i++) {
//     let y_pred = model(x)
//     let loss = mean_squared_error(y, y_pred)

//     model.cleargrads()
//     loss.backward()

//     optimizer.update()

//     if((i+1) % 100 === 0) {
//         console.log(loss.view)
//     }
// }

let arr = new Variable(Arr([[1,2,3], [4,5,6], [7,8,9]]))
console.log(arr.get([1,1]).view)
console.log(arr.slice(0, 2).view)