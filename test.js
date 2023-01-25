const {Variable} = require("./core")
const {} = require("./functions")
const Arr = require("./Arr")

let a = Arr([[[0], [1], [2]]])
console.log(a.squeeze([0, 2]).shape)