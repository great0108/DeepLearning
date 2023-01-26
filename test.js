const {Variable, sum} = require("./core")
const {} = require("./functions")
const Arr = require("./Arr")


let x0 = new Variable(Arr([1,2,3]))
let x1 = new Variable(Arr(10))
let y = x0.plus(x1)
console.log(y)

y.backward()
console.log(x0.grad)
console.log(x1.grad)