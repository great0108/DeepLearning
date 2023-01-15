function Callable() {
	let result = function() {
		return result.__call__.apply(result, arguments);
    }
	Object.setPrototypeOf(result, Callable.prototype)
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
	Object.setPrototypeOf(result, Callable2.prototype)
	return result;
}
Callable2.prototype.__proto__ = Callable.prototype
Callable2.prototype.__call__ = function(x) {
    return x+2
}