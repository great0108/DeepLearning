function A() {
    this = function() {return 1}
}

let a = new A()
console.log(a)
