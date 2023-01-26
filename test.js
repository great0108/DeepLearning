const {Variable, sum} = require("./core")
const {matmul} = require("./functions")
const Arr = require("./Arr")


let x = new Variable(Arr.range(6).reshape(2, 3))
let w = new Variable(Arr.range(12).reshape(3, 4))
let y = matmul(x, w)
y.backward()

console.log(x.grad)
console.log(w.grad)