const Arr = require("./Arr")

function a() {
    console.log(this)
    return function() {}
}

console.log(a())
console.log(new a())