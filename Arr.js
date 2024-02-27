(function() {

    "use strict"
    
    function deepCopy(target) {
        if(Array.isArray(target)) {
            let result = []
            for(let i = 0; i < target.length; i++) {
                result.push(deepCopy(target[i]))
            }
            target = null
            return result
        }
        if(typeof target === "object") {
            target = Object.assign({}, target)
            for(let a of Object.getOwnPropertyNames(target)) {
                target[a] = deepCopy(target[a])
            }
        }
        return target
    }
    
    function Arr(arr) {
        if(arr instanceof Arr) {
            arr = arr.copy()
        } else if(arguments.length > 1) {
            arr = Arr.copy(Array.from(arguments))
        } else if(Array.isArray(arr)) {
            arr = Arr.copy(arr)
        } else {
            arr = arr !== undefined ? [arr] : []
            Object.setPrototypeOf(arr, Arr.prototype)
        }
        return arr
    }
    
    Arr.prototype.__proto__ = Array.prototype
    
    Arr.fill = function(size, value) {
        size = Array.isArray(size) ? size : [size]
        size = size.slice()
        if(Array.isArray(value)) {
            value = Arr(value)
            let a = value.shape
            if(a.length > size.length || !a.same(size.slice(-a.length))) {
                throw new Error("지정한 크기에 값을 채울 수 없습니다.")
            }
            size = size.slice(0, -a.length)
        }
        size.reverse()
    
        for(let a of size) {
            value = Array(a).fill().map(v => deepCopy(value))
        }
        value = Arr.deepArr(value)
        return value
    }
    
    Arr.zeros = function(size) {
        size = arguments.length > 1 ? Array.from(arguments) : size
        return Arr.fill(size, 0)
    }
    
    Arr.rand = function(size) {
        size = arguments.length > 1 ? Array.from(arguments) : size
        return Arr.fill(size, 0).deepMap(v => Math.random())
    }
    
    Arr.range = function(start, end, step) {
        if(step === 0) {
            throw new Error("간격은 0이 될 수 없습니다.")
        }
        step = step === undefined ? 1 : step
        if(end === undefined) {
            end = start
            start = 0
        }
        let arr = []
        while(step > 0 ? start < end : start > end) {
            arr.push(start)
            start += step
        }
        Object.setPrototypeOf(arr, Arr.prototype)
        return arr
    }
    
    Arr.calShape = function(arr1, arr2) {
        let shape1 = arr1.shape.reverse()
        let shape2 = arr2.shape.reverse()
        let i = 0
        let shape = []
        while(shape1[i] !== undefined || shape2[i] !== undefined) {
            if(shape1[i] === undefined) {
                shape.push(shape2[i])
            } else if(shape2[i] === undefined) {
                shape.push(shape1[i])
            } else if(shape1[i] === 1 || shape2[i] === 1) {
                shape.push(shape1[i] * shape2[i])
            } else if(shape1[i] === shape2[i]) {
                shape.push(shape1[i])
            } else {
                throw new Error("크기가 맞지 않아서 계산할 수 없습니다.")
            }
            i++
        }
        return shape.reverse()
    }
    
    Arr.deepArr = function(arr) {
        if(Array.isArray(arr)) {
            if(!(arr instanceof Arr)) {
                Object.setPrototypeOf(arr, Arr.prototype)
            }
            for(let i = 0; i < arr.length; i++) {
                arr[i] = Arr.deepArr(arr[i])
            }
        }
        return arr
    }
    
    Arr.copy = function(arr) {
        arr = Arr.deepArr(deepCopy(arr))
        return arr
    }
    
    Object.defineProperty(Arr.prototype, "copy", {
        value : function() {
            return Arr.copy(this)
        }
    })
    
    Object.defineProperty(Arr.prototype, "same", {
        value : function(arr) {
            if(this.length !== arr.length) {
                return false
            }
            for(let i = 0; i < this.length; i++) {
                if(this[i] !== arr[i]) {
                    return false
                }
            }
            return true
        }
    })
    
    Object.defineProperty(Arr.prototype, "shape", {
        get : function() {
            let length = this.length
            if(length === 0) {
                return Arr(0)
            }
    
            let check = Array.isArray(this[0])
            for(let i = 1; i < length; i++) {
                if(Array.isArray(this[i]) !== check) {
                    throw new Error("크기가 일정하지 않습니다.")
                }
            }
            
            if(!check) {
                return Arr(length)
            }
    
            let size = this[0].shape
            for(let i = 1; i < length; i++) {
                if(!size.same(this[i].shape)) {
                    throw new Error("크기가 일정하지 않습니다.")
                }
            }
            size.unshift(length)
            return size
        }
    })
    
    Object.defineProperty(Arr.prototype, "size", {
        get : function() {
            return this.shape.reduce((a, v) => a*v, 1)
        }
    })
    
    Object.defineProperty(Arr.prototype, "ndim", {
        get : function() {
            return this.shape.length
        }
    })
    
    Object.defineProperty(Arr.prototype, "view", {
        get : function() {
            return this._view()
        }
    })
    
    Object.defineProperty(Arr.prototype, "_view", {
        value : function(ndim, n) {
            ndim = ndim === undefined ? this.ndim : ndim
            n = n === undefined ? ndim : n
            let result = ""
            if(ndim == 1) {
                for(let i = 0; i < this.length; i++) {
                    result += typeof this[i] === "object" ? JSON.stringify(this[i]) : this[i]
                    if(i !== this.length-1) {
                        result += ", "
                    }
                }
                return "[" + result + "]"
            }
            ndim -= 1
            for(let i = 0; i < this.length; i++) {
                result += this[i]._view(ndim, n)
                if(i !== this.length-1) {
                    result += "\n".repeat(ndim) + " ".repeat(n-ndim)
                }
            }
            return "[" + result + "]"
        }
    })
    
    
    Object.defineProperty(Arr.prototype, "get", {
        value : function(index) {
            index = arguments.length === 1 ?
            Array.isArray(index) ? index : [index] :
            Array.from(arguments)
            let temp = this
            for(let i of index) {
                temp = temp[i >= 0 ? i : temp.length + i]
            }
            return temp
        }
    })
    
    Object.defineProperty(Arr.prototype, "_get", {
        value : function(index) {
            let temp = this
            for(let i = 0; i < index.length; i++) {
                temp = temp[index[i]]
            }
            return temp
        }
    })
    
    Object.defineProperty(Arr.prototype, "set", {
        value : function(index, value) {
            index = Array.isArray(index) ? index.slice() : [index]
            let last = index.pop()
            let temp = this
            for(let i of index) {
                temp = temp[i >= 0 ? i : temp.length + i]
            }
            temp[last >= 0 ? last : temp.length + last] = Array.isArray(value) ? Arr(value) : value
        }
    })
    
    Object.defineProperty(Arr.prototype, "_set", {
        value : function(index, value) {
            let temp = this
            for(let i = 0; i < index.length-1; i++) {
                temp = temp[index[i]]
            }
            temp[index[index.length-1]] = value
        }
    })
    
    Object.defineProperty(Arr.prototype, "deepFor", {
        value : function(fn, index, size) {
            size = size === undefined ? this.shape : size
            index = index === undefined ? [] : index
            if(size.length-1 === index.length) {
                index.push(0)
                for(let i = 0; i < size[size.length-1]; i++) {
                    fn(this[i], index.slice())
                    index[index.length-1] += 1
                }
            } else {
                for(let i = 0; i < size[index.length]; i++) {
                    this[i].deepFor(fn, index.concat(i), size)
                }
            }
        }
    })
    
    Object.defineProperty(Arr.prototype, "deepMap", {
        value : function(fn, index, size) {
            size = size === undefined ? this.shape : size
            index = index === undefined ? [] : index
            index.push(0)
            if(size.length === 1) {
                let result = []
                for(let i = 0; i < this.length; i++) {
                    index[index.length-1] = i
                    result.push(fn(this[i], index.slice()))
                }
                Object.setPrototypeOf(result, Arr.prototype)
                return result
            }
    
            let result = []
            let length = size.shift()
            for(let i = 0; i < length; i++) {
                index[index.length-1] = i
                result.push(this[i].deepMap(fn, index.slice(), size.slice()))
            }
            Object.setPrototypeOf(result, Arr.prototype)
            return result
        }
    })
    
    Object.defineProperty(Arr.prototype, "_slice", {
        value : function(start, end) {
            let arr = []
            for(let i = start; i < end; i++) {
                arr.push(this[i])
            }
            Object.setPrototypeOf(arr, Arr.prototype)
            return arr
        }
    })
    
    Object.defineProperty(Arr.prototype, "slice", {
        value : function(start, end) {
            start = start === undefined ? 0 : start
            end = end === undefined ? this.length : end
            start = Array.isArray(start) ? start : [start]
            end = Array.isArray(end) ? end : [end]
            while(start.length < end.length) {
                start.push(0)
            }
            while(start.length > end.length) {
                end.push(-0.5)
            }
    
            let size = this.shape
            start = start.map((v, i) => v >= 0 ? v : size[i] + v)
            start = start.map((v, i) => Math.min(Math.max(0, v), size[i]))
            end = end.map((v, i) => v >= 0 ? v : size[i] + v)
            end = end.map((v, i) => Math.min(Math.max(0, v), size[i]))
    
            let arr = this._slice(start.shift(), end.shift())
            if(start.length > 0) {
                for(let i = 0; i < arr.length; i++) {
                    arr[i] = arr[i].slice(start.slice(), end.slice())
                }
            }
            return arr
        }
    })
    
    Object.defineProperty(Arr.prototype, "overlap", {
        value : function(value, start, end) {
            start = start === undefined ? [] : start
            start = Array.isArray(start) ? start : [start]
            end = end === undefined ? [] : end
            end = Array.isArray(end) ? end : [end]
        
            let size = this.shape
            while(start.length < size.length) {
                start.push(0)
            }
            while(end.length < size.length) {
                end.push(size[end.length])
            }
        
            start = start.map((v, i) => v >= 0 ? v : size[i] + v)
            start = start.map((v, i) => Math.min(Math.max(0, v), size[i]))
            end = end.map((v, i) => v >= 0 ? v : size[i] + v)
            end = end.map((v, i) => Math.min(Math.max(0, v), size[i]))
            
            let shape = start.map((v, i) => end[i] - v)
            value = Arr.deepArr(value).broadcast(shape)
            value.deepFor((v, i) => this._set(i.map((v2, i2) => v2 + start[i2]), v))
        }
    })
    
    Object.defineProperty(Arr.prototype, "expand", {
        value : function(axis) {
            axis = arguments.length === 1 ? axis : Array.from(arguments)
            if(Array.isArray(axis)) {
                let temp = this
                for(let a of axis.sort((a, b) => a - b)) {
                    temp = temp.expand(a)
                }
                return temp
            }
            let ndim = this.ndim
            axis = axis >= 0 ? axis : ndim + axis + 1
            if(ndim < axis || axis < 0) {
                throw new Error("차원이 배열을 벗어났습니다.")
            }
            return Arr.deepArr(this._expand(axis))
        }
    })
    
    Object.defineProperty(Arr.prototype, "_expand", {
        value : function(axis) {
            if(axis == 0) {
                return [this]
            } else if(axis == 1) {
                return this.map(v => [v])
            } else {
                return this.map(v => v._expand(axis-1))
            }
        }
    })
    
    Object.defineProperty(Arr.prototype, "squeeze", {
        value : function(axis) {
            if(axis === undefined) {
                let temp = this
                while(Array.isArray(temp) && temp.shape.includes(1)) {
                    temp = temp.squeeze(temp.shape.indexOf(1))
                }
                return temp
            }
            axis = arguments.length === 1 ? axis : Array.from(arguments)
            if(Array.isArray(axis)) {
                let temp = this
                for(let a of axis.sort((a, b) => b - a)) {
                    temp = temp.squeeze(a)
                }
                return temp
            }
            let size = this.shape
            axis = axis >= 0 ? axis : size.length + axis
            if(size.length <= axis || axis < 0) {
                throw new Error("차원이 배열을 벗어났습니다.")
            } else if(size[axis] != 1) {
                throw new Error("배열의 길이가 1이 아니라서 차원을 축소할 수 없습니다.")
            }
            return Arr.copy(this._squeeze(axis))
        }
    })
    
    Object.defineProperty(Arr.prototype, "_squeeze", {
        value : function(axis) {
            if(axis == 0) {
                return this[0]
            } else if(axis == 1) {
                return this.map(v => v[0])
            } else {
                return this.map(v => v._squeeze(axis-1))
            }
        }
    })
    
    Object.defineProperty(Arr.prototype, "broadcast", {
        value : function(shape) {
            shape = arguments.length === 1 
            ? (Array.isArray(shape) ? shape : [shape]) 
            : Array.from(arguments)
    
            let size = this.shape
            if(size.same(shape)) {
                return this
            }
    
            size = size.reverse()
            shape = shape.slice().reverse()
            let temp = this
            for(let i = 0; i < shape.length; i++) {
                if(size[i] === undefined) {
                    temp = [temp]
                    Object.setPrototypeOf(temp, Arr.prototype)
                } else if(size[i] !== 1 && size[i] !== shape[i]) {
                    throw new Error("크기가 맞지 않습니다.")
                }
            }
            return Arr.copy(temp._broadcast(shape.reverse()))
        }
    })
    
    Object.defineProperty(Arr.prototype, "_broadcast", {
        value : function(shape) {
            if(shape.length === 1) {
                return this.length === 1 ? Array(shape[0]).fill(this[0]) : this.slice()
            }
            let a = shape.shift()
            let temp = []
            for(let i = 0; i < this.length; i++) {
                temp.push(this[i]._broadcast(shape.slice()))
            }
            if(a === temp.length) {
                return temp
            }
            let arr = []
            for(let i = 0; i < a; i++) {
                arr.push(deepCopy(temp[0]))
            }
            return arr
        }
    })
    
    Object.defineProperty(Arr.prototype, "cal", {
        value : function(arr, fn) {
            if(!(arr instanceof Arr)) {
                arr = Arr(arr)
            }
            let shape = Arr.calShape(this, arr)
            let arr1 = this.broadcast(shape)
            let arr2 = arr.broadcast(shape)
            return arr1.deepMap((v, i) => fn(v, arr2._get(i)))
        }
    })
    
    Object.defineProperty(Arr.prototype, "plus", {
        value : function(arr) {
            return this.cal(arr, (a, b) => a+b)
        }
    })
    
    Object.defineProperty(Arr.prototype, "minus", {
        value : function(arr) {
            return this.cal(arr, (a, b) => a-b)
        }
    })
    
    Object.defineProperty(Arr.prototype, "rminus", {
        value : function(arr) {
            return this.cal(arr, (a, b) => b-a)
        }
    })
    
    Object.defineProperty(Arr.prototype, "mul", {
        value : function(arr) {
            return this.cal(arr, (a, b) => a*b)
        }
    })
    
    Object.defineProperty(Arr.prototype, "div", {
        value : function(arr) {
            return this.cal(arr, (a, b) => a/b)
        }
    })
    
    Object.defineProperty(Arr.prototype, "rdiv", {
        value : function(arr) {
            return this.cal(arr, (a, b) => b/a)
        }
    })
    
    Object.defineProperty(Arr.prototype, "pow", {
        value : function(arr) {
            return this.cal(arr, (a, b) => Math.pow(a, b))
        }
    })
    
    Object.defineProperty(Arr.prototype, "flat", {
        value : function(deep) {
            deep = deep === undefined ? 100 : deep
            let result = []
            let temp = this
            for(let i = 0; i < deep+1; i++) {
                if(!Array.isArray(temp[0])) {
                    break
                }
                result = []
                for(let j = 0; j < temp.length; j++) {
                    for(let k = 0; k < temp[j].length; k++) {
                        result.push(temp[j][k])
                    }
                }
                temp = result
            }
            result = null
            return Arr.copy(temp)
        }
    })
    
    Object.defineProperty(Arr.prototype, "reshape", {
        value : function(size) {
            size = arguments.length === 1 
            ? (Array.isArray(size) ? size : [size]) 
            : Array.from(arguments)
            let temp = this.flat()
            if(size.includes(-1)) {
                if(size.filter(v => v == -1).length != 1) {
                    throw new Error("-1은 하나만 사용할 수 있습니다.")
                }
                let index = size.indexOf(-1)
                size.splice(index, 1)
                let tempSize = (temp.length / size.reduce((a, v) => a*v, 1)) | 0
                size.splice(index, 0, tempSize)
            }
            if(temp.length != size.reduce((a, v) => a*v, 1)) {
                throw new Error("배열을 지정한 모양으로 바꿀 수 없습니다.")
            }
            let index = 0
            let result = Arr.zeros(size)
            result.deepFor((v, i) => {
                result._set(i, temp[index])
                index += 1
            }, [], size)
            temp = null
            return result
        }
    })
    
    Object.defineProperty(Arr.prototype, "swapaxis", {
        value : function(axis1, axis2, size) {
            if(axis1 === axis2) {
                return this
            }
            size = size === undefined ? this.shape : size
            axis1 = axis1 >= 0 ? axis1 : size.length + axis1
            axis2 = axis2 >= 0 ? axis2 : size.length + axis2
            if(axis1 >= size.length || axis1 < 0) {
                throw new Error("차원이 배열을 벗어났습니다.")
            } else if(axis2 >= size.length || axis2 < 0) {
                throw new Error("차원이 배열을 벗어났습니다.")
            }
            let size2 = size.slice()
            size2[axis1] = size[axis2]
            size2[axis2] = size[axis1]
         
            size2[size2.length-1] = 1
            let result = Arr.zeros(size2)
            this.deepFor((v, i) => {
                let temp = i[axis1]
                i[axis1] = i[axis2]
                i[axis2] = temp
                result._set(i, v)
            }, [], size)
            return result
        }
    })
    
    Object.defineProperty(Arr.prototype, "transpose", {
        value : function(axis) {
            if(axis === undefined) {
                return this.T
            }
    
            axis = arguments.length === 1 
            ? (Array.isArray(axis) ? axis : [axis]) 
            : Array.from(arguments)
    
            let size = this.shape
            if(axis.length != size.length) {
                throw new Error("차원 개수가 맞지 않습니다.")
            }
            axis = axis.map(v => v >= 0 ? v : size.length + v)
    
            for(let i = 0; i < size.length; i++) {
                if(!axis.includes(i)) {
                    throw new Error("없는 차원이 있습니다.")
                }
            }
    
            let size2 = size.slice()
            for(let i = 0; i < size.length; i++) {
                size2[i] = size[axis[i]]
            }
    
            let result = Arr.zeros(size2)
            this._transpose(result, axis.map((v, i) => axis.indexOf(i)))
            return result
        }
    })
    
    Object.defineProperty(Arr.prototype, "_transpose", {
        value : function(arr, axis, index, size) {
            size = size === undefined ? this.shape : size
            index = index === undefined ? Array(size.length).fill().map(v => 0) : index
            let nowaxis = index.length - size.length
    
            if(size.length === 1) {
                for(let i = 0; i < size[0]; i++) {
                    index[axis[nowaxis]] = i
                    arr._set(index, this[i])
                }
            } else {
                let iter = size[0]
                size = size.slice(1)
                for(let i = 0; i < iter; i++) {
                    index[axis[nowaxis]] = i
                    this[i]._transpose(arr, axis, index, size)
                }
            }
        }
    })
    
    Object.defineProperty(Arr.prototype, "_calaxis", {
        value : function(fn, axis, keepdims) {
            let result = this
            let size = this.shape
            axis = axis >= 0 ? axis : size.length + axis
    
            if(size.length-1 != axis) {
                let order = Array(size.length).fill().map((v, i) => i)
                order.splice(axis, 1)
                order.push(axis)
                result = result.transpose(order)
            }
    
            size.splice(axis, 1)
            let arr = Arr.zeros(size)
            arr = arr.deepMap((v, i) => {
                v = fn(result._get(i), i)
                if(Array.isArray(v)) {
                    Object.setPrototypeOf(v, Arr.prototype)
                }
                return v
            })
    
            let arrSize = arr.shape
    
            if(arrSize.length > size.length) {
                let order = Array(arrSize.length).fill().map((v, i) => i)
                order.splice(axis, 0, order.length-1)
                order.pop()
                arr = arr.transpose(order)
            } else if(keepdims) {
                arr = arr.expand(axis)
            }
            
            return arr
        }
    })
    
    Object.defineProperty(Arr.prototype, "calaxis", {
        value : function(fn, axis, keepdims, opt) {
            if(axis === undefined || axis === null) {
                let ndim = this.ndim
                if(opt) {
                    let result = [fn(this.flat())]
                    if(keepdims) {
                        for(let i = 1; i < ndim; i++) {
                            result = [result]
                        }
                    }
                    result = Arr.deepArr(result)
                    return result
                }
                axis = Array(ndim).fill().map((v, i) => i)
            }
    
            if(Array.isArray(axis)) {
                let result = this
                for(let a of axis) {
                    result = result.calaxis(fn, a, keepdims)
                }
                return result
            }
    
            let result = this._calaxis(fn, axis, keepdims)
            return result
        }
    })
    
    Object.defineProperty(Arr.prototype, "max", {
        value : function(axis, keepdims) {
            return this.calaxis(v => Math.max.apply(null, v), axis, keepdims, true)
        }
    })
    
    Object.defineProperty(Arr.prototype, "argmax", {
        value : function(axis, keepdims) {
            return this.calaxis(v => v.indexOf(Math.max.apply(null, v)), axis, keepdims)
        }
    })
    
    Object.defineProperty(Arr.prototype, "min", {
        value : function(axis, keepdims) {
            return this.calaxis(v => Math.min.apply(null, v), axis, keepdims, true)
        }
    })
    
    Object.defineProperty(Arr.prototype, "sum", {
        value : function(axis, keepdims) {
            return this.calaxis(v => v.reduce((a, v) => a+v, 0), axis, keepdims, true)
        }
    })
    
    Object.defineProperty(Arr.prototype, "T", {
        get : function() {
            let ndim = this.ndim
            if(ndim == 1) {
                return this.slice()
            }
            let temp = this
            for(let i = 0; i < (ndim-1)/2; i++) {
                temp = temp.swapaxis(i, ndim-i-1)
            }
            return temp
        }
    })
    
    Object.defineProperty(Arr.prototype, "flip", {
        value : function(axis) {
            axis = axis === undefined ? 0 : axis
            axis = arguments.length === 1 ? axis : Array.from(arguments)
    
            if(Array.isArray(axis)) {
                let temp = this
                for(let a of axis) {
                    temp = temp.flip(a)
                }
                return temp
            }
            if(axis === undefined || axis === 0) {
                return this._flip()
            }
            let size = this.shape
            axis = axis >= 0 ? axis : size.length + axis
            if(size.length <= axis || axis < 0) {
                throw new Error("차원이 배열을 벗어났습니다.")
            }
            let arr = Arr.zeros(size.slice(0, axis))
            arr = arr.deepMap((v, i) => this._get(i)._flip())
            return arr
    
        }
    })
    
    Object.defineProperty(Arr.prototype, "_flip", {
        value : function() {
            const len = this.length
            let result = []
            for(let i = 0; i < len; i++) {
                result.push(this[len-i-1])
            }
            Object.setPrototypeOf(result, Arr.prototype)
            return result
        }
    })
    
    Object.defineProperty(Arr.prototype, "select", {
        value : function(index, axis) {
            axis = axis === undefined ? 0 : axis
            let size = this.shape
            axis = axis >= 0 ? axis : size.length + axis
            if(size.length <= axis || axis < 0) {
                throw new Error("차원이 배열을 벗어났습니다.")
            }
            
            index = Array.isArray(index) ? index : [index]
            index = index.map(v => v >= 0 ? v : size[axis] + v)
            for(let i of index) {
                if(size[axis] <= i || index < 0) {
                    throw new Error("인덱스가 배열을 벗어났습니다.")
                }
            }
            size[axis] = index.length
            let arr = Arr.zeros(size)
            arr = arr.deepMap((v, i) => {
                i[axis] = index[i[axis]]
                return this._get(i)
            })
            return arr
        }
    })
    
    Object.defineProperty(Arr.prototype, "_matmul", {
        value : function(arr, shape1, shape2) {
            let result = Arr.zeros(shape1[0], shape2[1])
            for(let k = 0; k < shape1[1]; k++) {
                for(let i = 0; i < shape1[0]; i++) {
                    for(let j = 0; j < shape2[1]; j++) {
                        result[i][j] += arr[k][j] * this[i][k]
                    }
                }
            }
            arr = null
            return result
        }
    })
    
    Object.defineProperty(Arr.prototype, "matmul", {
        value : function(arr2) {
            if(!(arr2 instanceof Arr)) {
                arr2 = Arr(arr2)
            }
            let arr1 = this
            let shape1 = arr1.shape
            let shape2 = arr2.shape
            let ndim1 = shape1.length
            let ndim2 = shape2.length
    
            if(shape1.length === 1) {
                arr1 = arr1.expand(0)
                shape1.unshift(1)
            }
            if(shape2.length === 1) {
                arr2 = arr2.expand(-1)
                shape2.push(1)
            }
    
            if(shape1.length !== shape2.length) {
                throw new Error("두 배열의 차원이 다릅니다.")
            } else if(shape1[shape1.length-1] !== shape2[shape2.length-2]) {
                throw new Error("행과 열의 길이가 다릅니다.")
            } else if(!shape1.slice(0, -2).same(shape2.slice(0, -2)) ) {
                throw new Error("배치 차원이 다릅니다.")
            }
    
            let result = null
            if(shape1.length === 2) {
                result = arr1._matmul(arr2, shape1, shape2)
            } else {
                let s1 = shape1.slice(-2)
                let s2 = shape2.slice(-2)
                result = Arr.zeros(shape1.slice(0, -2))
                result = result.deepMap((v, i) => arr1._get(i)._matmul(arr2._get(i), s1, s2))
            }
            result = Arr.deepArr(result)
            if(ndim1 === 1) {
                result = result.squeeze(0)
            }
            if(ndim2 === 1) {
                result = result.squeeze(-1)
            }
            return result
        }
    })
    
    Object.defineProperty(Arr.prototype, "shuffle", {
        value : function() {
            let arr = this.slice()
            const result = Arr.zeros(arr.length)
            for(let i = arr.length-1; i >= 0 ; i--) {
                let j = Math.random() * (i+1) | 0
                result[i] = arr[j]
                arr[j] = arr[i]
            }
            return result
        }
    })
    
    Object.defineProperty(Arr.prototype, "choose", {
        value : function(index, axis) {
            let arr = this
            let size = arr.shape
    
            index = index instanceof Arr ? index : Arr(index)
            let indexSize = index.shape
            if(size.length > indexSize.length) {
                throw new Error("인덱스 차원이 배열 차원과 같거나 커야합니다.")
            } else if(size.length < indexSize.length) {
                size = indexSize.slice(0, -size.length).concat(size)
                arr = arr.broadcast(size)
            }
    
            axis = axis === undefined ? -1 : axis
            axis = axis >= 0 ? axis : size.length + axis
            if(size.length <= axis || axis < 0) {
                throw new Error("차원이 배열을 벗어났습니다.")
            }
            index = index.deepMap(v => v >= 0 ? v : size[axis] + v)
    
            let result = Arr.zeros(indexSize)
            result = index.deepMap((v, i) => {
                i.splice(axis, 0, v)
                i.pop()
                return arr._get(i)
            })
            return result
        }
    })
    
    Object.defineProperty(Arr.prototype, "cat", {
        value : function(arr, axis, multi) {
            axis = axis === undefined ? 0 : axis
            let size = this.shape
            axis = axis >= 0 ? axis : size.length + axis
            if(size.length <= axis || axis < 0) {
                throw new Error("차원이 배열을 벗어났습니다.")
            }
    
            if(!multi) {
                arr = [arr]
            }
    
            let axisLength = size[axis]
            let arrSize = []
            for(let i = 0; i < arr.length; i++) {
                arr[i] = arr[i] instanceof Arr ? arr[i] : Arr(arr[i])
                arrSize.push(arr[i].shape)
                axisLength += arrSize[i][axis]
                
                if(!size.every((v, j) => j == axis || v == arrSize[i][j])) {
                    throw new Error("배열 크기가 달라서 합칠 수 없습니다.")
                }
            }
    
            arrSize.unshift(size)
            arr.unshift(this)
    
            let resultSize = size.slice()
            resultSize[axis] = axisLength
            let result = Arr.zeros(resultSize)
            for(let i = 0; i < arrSize.length; i++) {
                let start = 0
                for(let j = 0; j < i; j++) {
                    start += arrSize[j][axis]
                }
    
                arr[i].deepFor((v, i) => {
                    i[axis] += start
                    result._set(i, v)
                })
            }
            return result
        }
    })
    
    Object.defineProperty(Arr.prototype, "insert", {
        value : function(index, value, axis) {
            // if(Array.isArray(index)) {
            //     for(let i = 0; i < index.length; i++) {
            //         this.insert(index[i], value, axis)
            //     }
            //     return this
            // }
        
            axis = axis === undefined ? 0 : axis
            let size = this.shape
            axis = axis >= 0 ? axis : size.length + axis
            if(size.length <= axis || axis < 0) {
                throw new Error("차원이 배열을 벗어났습니다.")
            }
        
            value = value instanceof Arr ? value : Arr(value)
            let valueSize = value.shape
            let newSize = size.slice()
            if(valueSize.length == size.length) {
                newSize[axis] = valueSize[axis]
            } else {
                newSize[axis] = 1
            }
            value = value.broadcast(newSize)
        
            if(axis === 0) {
                Arr.prototype.splice.apply(this, [index, 0].concat(value.copy()))
                return this
            }

            index = index instanceof Arr ? index : Arr(index)
            index = index.broadcast(size.slice(0, axis))
            
            index.deepFor((v, i) => {
                Arr.prototype.splice.apply(this._get(i), [v, 0].concat(value._get(i).copy()))
            })
            return this
        }
    })
    
    Object.defineProperty(Arr.prototype, "delete", {
        value : function(index, count, axis) {
            if(Array.isArray(index)) {
                let result = []
                for(let i = 0; i < index.length; i++) {
                    result.push(this.delete(index[i]-count*i, count, axis))
                }
        
                if(result.length == 1) {
                    return result[0]
                }
        
                let temp = result.shift()
                return temp.cat(result, axis, true)
            }
        
            axis = axis === undefined ? 0 : axis
            let size = this.shape
            axis = axis >= 0 ? axis : size.length + axis
            if(size.length <= axis || axis < 0) {
                throw new Error("차원이 배열을 벗어났습니다.")
            }
    
            count = count === undefined ? size[axis] : count
            if(axis === 0) {
                return this.splice(index, count)
            }
            
            let arr = Arr.zeros(size.slice(0, axis))
            arr = arr.deepMap((v, i) => {
                return this._get(i).splice(index, count).copy()
            })
            return arr
        }
    })
    
    Object.defineProperty(Arr.prototype, "repeat", {
        value : function(repeats, axis) {
            axis = axis === undefined ? -1 : axis
            let size = this.shape
            axis = axis >= 0 ? axis : size.length + axis
            if(size.length <= axis || axis < 0) {
                throw new Error("차원이 배열을 벗어났습니다.")
            }

            repeats = Array.isArray(repeats) ? repeats : [repeats]
            if(!repeats.every(v => v > 0)) {
                throw new Error("반복 횟수는 양수여야합니다.")
            }
            repeats = repeats.concat(Array(size.length - axis-1).fill(1))

            let newSize = size.slice()
            newSize.reverse()
            repeats.reverse()
            for(let i = 0; i < repeats.length; i++) {
                if(newSize.length > i) {
                    newSize[i] *= repeats[i]
                } else {
                    newSize.push(repeats[i])
                }
            }
    
            newSize.reverse()
            let arr = Arr.zeros(newSize)
            arr = arr.deepMap((v, i) => {
                let index = i.slice(-size.length).map((v, j) => v % size[j])
                return this._get(index)
            })
            return arr
        }
    })
    
    
    // overriding
    
    Object.defineProperty(Arr.prototype, "map", {
        value : function(fn, thisarg) {
            let arr = Array.prototype.map.call(this, fn, thisarg)
            Object.setPrototypeOf(arr, Arr.prototype)
            return arr
        }
    })
    
    Object.defineProperty(Arr.prototype, "filter", {
        value : function(fn, thisarg) {
            let arr = Array.prototype.filter.call(this, fn, thisarg)
            Object.setPrototypeOf(arr, Arr.prototype)
            return arr
        }
    })
    
    Object.defineProperty(Arr.prototype, "splice", {
        value : function() {
            let arr = Array.prototype.splice.apply(this, Array.from(arguments))
            Object.setPrototypeOf(arr, Arr.prototype)
            return arr
        }
    })
    
    Object.defineProperty(Arr.prototype, "concat", {
        value : function() {
            let arr = Array.prototype.concat.apply(this, Array.from(arguments))
            Object.setPrototypeOf(arr, Arr.prototype)
            return arr
        }
    })
    
    Object.defineProperty(Arr.prototype, "reverse", {
        value : function() {
            Array.prototype.reverse.call(this)
            Object.setPrototypeOf(this, Arr.prototype)
            return this
        }
    })
    
    Object.defineProperty(Arr.prototype, "sort", {
        value : function(fn) {
            Array.prototype.sort.call(this, fn)
            Object.setPrototypeOf(this, Arr.prototype)
            return this
        }
    })
    
    
    // function test() {
    //     let a = Arr.zeros(5,5,5)
    //     let start = Date.now()
    //     for(let i = 0; i < 1e5; i++) {
    //         a.transpose([2,0,1])
    //     }
    //     console.log(Date.now() - start)
    // }

    let a = Arr([[1,2], [3,4]])
    console.log(a.insert([0,1], [[5], [6]], 1))

    // let b = Arr([1,2,3,4])
    // console.log(b.insert(0, [5,6]))
    
    
    module.exports = Arr
    
    })()