function Callable() {
    let self = this
    console.log(self)
	let result = function Callable() {
		return self.__call__.apply(self, arguments);
    }
	result.__proto__ = Callable.prototype;
	return result;
}
Callable.prototype.__call__ = function(x) {
    return x+1
}
Callable.prototype.a = function() {
    return 1
}

function Callable2() {
    let result = Callable.call(this)
	result.__proto__ = Callable2.prototype;
	return result;
}
Callable2.prototype.__proto__ = Callable.prototype
Callable2.prototype.__call__ = function(x) {
    return x+2
}

let a = new Callable2()
console.log(a(1))
