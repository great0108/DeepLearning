const Arr = require("./Arr")

function a() {
    return Array.from(arguments)
}

let b = {a:1}
let c = a(b, b)
console.log(c[0] === c[1])
