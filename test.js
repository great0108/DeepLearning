const {Variable, no_grad} = require("./core")
const Arr = require("./Arr")

let x = new Variable(Arr(3))
let y = x.add(3).pow(2)
y.backward()


console.log(y)
console.log(x)