const {Variable, Operation} = require("./core")
const {sin, tanh} = require("./functions")
const Arr = require("./Arr")

let x = new Variable(Arr([[1,2,3], [4,5,6]]))
let c = new Variable(Arr([[10, 20, 30], [40, 50, 60]]))

let y = x.mul(c).div(10)
y.backward()
console.log(x.grad)
console.log(c.grad)