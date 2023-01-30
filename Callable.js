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
		let result = Callable.inherit(Callable2)
		result.b = 1
		return result
	}
	Callable2.inherit = Callable.inherit
	Callable2.prototype.__proto__ = Callable.prototype
	Callable2.prototype.__call__ = function(x) {
		return x+2
	}

	function Callable3() {
		return Callable2.inherit(Callable3)
	}
	Callable3.prototype.__proto__ = Callable2.prototype
	Callable3.prototype.__call__ = function(x) {
		return x+2
	}

	module.exports = Callable
})()