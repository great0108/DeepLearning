function as_array(x) {
    if(typeof x === "number") {
        return Array(x)
    }
    return x
}

console.log(as_array(1))
console.log(as_array([1]))
console.log(as_array(null))