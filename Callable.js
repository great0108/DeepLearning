(function() {
	"use strict"

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
	Callable.inherit = function(child) {
		let result = this()
		Object.setPrototypeOf(result, child.prototype)
		return result
	}
	
	function Callable2() {
		return Callable.inherit(Callable2)
	}
	Callable2.prototype.__proto__ = Callable.prototype
	Callable2.prototype.__call__ = function(x) {
		return x+2
	}

	module.exports = Callable
})()