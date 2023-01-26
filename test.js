const {Variable, sum} = require("./core")
const {} = require("./functions")
const Arr = require("./Arr")


let x = new Variable(Arr([[1,2,3],[4,5,6]]))
let y = sum(x, null, true)
y.backward()
console.log(y)
console.log(x.grad)