const Arr = require("./Arr")
function List(arr) {
    arr.__proto__ = List.prototype
    return arr
}
List.prototype.__proto__ = Array.prototype

let a = Arr()
console.log(a instanceof List)
a = List(a)
console.log(a instanceof List)