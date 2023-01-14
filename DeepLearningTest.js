const {Arr, Arr_prototype} = require("../게임공간 모듈/Arr(prototype ver).js")
Array.__proto__ = Object.assign(Array.__proto__, Arr)
Array.prototype = Object.assign(Array.prototype, Arr_prototype)

function copy(value) {
    if(typeof value === "object") {
        return JSON.parse(JSON.stringify(value))
    }
    return value
}

function matmul(arr1, arr2) {
    return arr1.map((row) => arr2[0].map((_,y) => row.reduce((a,b,c) => a + b * arr2[c][y], 0)))
}

function trans(arr) {
    return arr[0].map((_, y) => arr.map((_, x) => arr[x][y]))
}

function softmax(x) {
    if(typeof x[0] == "number") {
        let max = Math.max.apply(null, x)
        x = x.map(v => Math.exp(v-max))
        let sum = 0
        for(let i = 0; i < x.length; i++) {
            sum += x[i]
        }
        return x.map(v => v/sum)
    } else {
        let arr = []
        for(let i = 0; i < x.length; i++) {
            arr.push(softmax(x[i]))
        }
        return arr
    }
}

function mean_squared_error(y, t) {
    if(typeof y[0] == "number") {
        let sum = 0
        for(let i = 0; i < y.length; i++) {
            sum += Math.pow(y-t, 2)
        }
        return sum
    } else {
        let sum = 0
        for(let i = 0; i < y.length; y++) {
            sum += mean_squared_error(y[i], t[i])
        }
        return sum/y.length
    }
}

function cross_entropy_error(y, t) {
    if(typeof y[0] == "number") {
        if(typeof t != "number") {
            let max = Math.max.apply(null, t)
            t = t.indexOf(max)
        }
        return Math.log(y[t] + 1e-7)
    } else {
        let sum = 0
        for(let i = 0; i < y.length; i++) {
            sum += cross_entropy_error(y[i], t[i])
        }
        return -sum/y.length
    }
}

function gaussRandom() {
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    /*if outside interval [0,1] start over*/
    if(r == 0 || r > 1) return gaussRandom();

    var c = Math.sqrt(-2*Math.log(r)/r);
    return Math.random() > 0.5 ? u*c : -u*c;
}

function load_data() {
    let N = 300  //클래스당 샘플 수
    let DIM = 2  //데이터 요소 수
    let CLS_NUM = 3  //클래스 수
    
    let x = Array(N*CLS_NUM).fill().map(v => Array(DIM).fill(0))
    let t = Array(N*CLS_NUM).fill().map(v => Array(CLS_NUM).fill(0))

    for(let j = 0; j < CLS_NUM; j++) {
        for(let i = 0; i < N; i++) {
            let rate = (i+1) / N
            let theta = j*4 + 4*rate + gaussRandom()*0.2

            let ix = N*j + i
            x[ix] = [rate*Math.sin(theta), rate*Math.cos(theta)]
            t[ix][j] = 1
        }
    }
    return [x,t]
}

function Sigmoid() {
    this.params = []
    this.grads = []
    this.out = null
}
Sigmoid.prototype.forward = function(x) {
    let out = x.map(v => v.map(v2 => 1 / (1 + Math.exp(-v2))))
    this.out = out
    return out
}
Sigmoid.prototype.backward = function(dout) {
    return this.out.map((v, i) => v.map((v2, j) => dout[i][j] * (1-v2) * v2))
}

function Relu() {
    this.params = []
    this.grads = []
    this.out = null
}
Relu.prototype.forward = function(x) {
    let out = x.map(v => v > 0 ? v : 0)
    this.out = out
    return out
}
Relu.prototype.backward = function(dout) {
    return this.out.map((v, i) => v > 0 ? dout[i] : 0)
}

function SoftmaxWithCEE() {
    this.params = []
    this.grads = []
    this.y = []
    this.t = []
}
SoftmaxWithCEE.prototype.forward = function(x, t) {
    this.t = t
    this.y = softmax(x)

    if(this.t[0].length === this.y[0].length) {
        this.t = this.t.map(v => v.indexOf(Math.max.apply(null, v)))
    }
    let loss = cross_entropy_error(this.y, this.t)
    return loss
}
SoftmaxWithCEE.prototype.backward = function(dout) {
    dout = dout === undefined ? 1 : dout
    let batch_size = this.t.length
    let dx = copy(this.y)
    for(let i = 0; i < batch_size; i++) {
        dx[i][this.t[i]] -= 1
    }
    return dx.map(v => v.map(v2 => v2 * dout / batch_size))
}

function Affine(W, b) {
    this.params = [W, b]
    this.grads = [0, 0]
    this.x = null
}
Affine.prototype.forward = function(x) {
    let W = this.params[0]
    let b = this.params[1]
    let out = matmul(x, W)
    out = out.map(v => v.map((v2, i) => v2+b[i]))
    this.x = x
    return out
}
Affine.prototype.backward = function(dout) {
    let W = this.params[0]
    let dx = matmul(dout, trans(W))
    let dW = matmul(trans(this.x), dout)

    let db = Array(dout[0].length).fill(0)
    for(let i = 0; i < dout[0].length; i++) {
        for(let j = 0; j < dout.length; j++) {
            db[i] += dout[j][i]
        }
    }

    this.grads[0] = dW
    this.grads[1] = db
    return dx
}

function SGD(lr) {
    this.lr = lr === undefined ? 0.01 : lr
}
SGD.prototype.update = function(params, grads) {
    for(let i = 0; i < params.length; i++) {
        if(typeof params[i][0] == "object" && params[i][0].length != params[i][1].length) {
            for(let j = 0; j < params[i].length; j++) {
                this._update(params[i][j], grads[i][j])
            }
        } else {
            this._update(params[i], grads[i])
        }
    }
}
SGD.prototype._update = function(p, g) {
    if(typeof p[0] == "number") {
        for(let j = 0; j < p.length; j++) {
            p[j] -= g[j] * this.lr
        }
    } else {
        for(let j = 0; j < p.length; j++) {
            for(let k = 0; k < p[0].length; k++) {
                p[j][k] -= g[j][k] * this.lr
            }
        }
    }
}

function Adam(lr, beta1, beta2) {
    this.lr = lr === undefined ? 0.001 : lr
    this.beta1 = beta1 === undefined ? 0.9 : beta1
    this.beta2 = beta2 === undefined ? 0.999 : beta2
    this.iter = 0
    this.m = null
    this.v = null
}
Adam.prototype.update = function(params, grads) {
    if(this.m == null) {
        this.m = []
        this.v = []
        for(param of params) {
            let zeros = []
            for(h of param) {
                zeros.push(Array.zeros(h.shape()))
            }
            this.m.push(copy(zeros))
            this.v.push(zeros)
        }
        console.log(this.m[0])
    }
    this.iter += 1
    let lr_t = this.lr * Math.sqrt(1.0 - Math.pow(this.beta2, this.iter)) / (1.9 - Math.pow(this.beta1, this.iter))

    for(let i = 0; i < params.length; i++) {
        Array.For(this.m[i], index => {
            this.m[i].put(index, this.m[i].get(index) + (1 - this.beta1) * (grads[i].get(index) - this.m[i].get(index)))
            this.v[i].put(index, this.v[i].get(index) + (1 - this.beta2) * (Math.pow(grads[i].get(index), 2) - this.v[i].get(index)))

            params[i].put(index, params[i].get(index) - (lr_t * this.m[i].get(index) / (Math.sqrt(this.v[i].get(index)) + 1e-7)))
        })
    }
}
Adam.prototype._zero = function(p) {
    if(typeof p[0] == "number") {
        return Array(p.length).fill(0)
    } else {
        return Array(p.length).fill().map(v => Array(p[0].length).fill(0))
    }
}

function TwoLayerNet(input_size, hidden_size, output_size) {
    let I = input_size, 
        H = hidden_size, 
        O = output_size

    let W1 = Array(I).fill().map(v => Array(H).fill().map(v2 => gaussRandom()*0.05))
    let b1 = Array(H).fill().map(v => gaussRandom()*0.05)
    let W2 = Array(H).fill().map(v => Array(O).fill().map(v2 => gaussRandom()*0.05))
    let b2 = Array(O).fill().map(v => gaussRandom()*0.05)

    this.layers = [
        new Affine(W1, b1),
        new Sigmoid(),
        new Affine(W2, b2)
    ]
    this.loss_layer = new SoftmaxWithCEE()

    this.params = []
    this.grads = []
    for(layer of this.layers) {
        if(layer.params.length === 0) continue
        this.params.push(layer.params)
        this.grads.push(layer.grads)
    }
}
TwoLayerNet.prototype.predict = function(x) {
    for(layer of this.layers) {
        x = layer.forward(x)
    }
    return x
}
TwoLayerNet.prototype.forward = function(x, t) {
    let score = this.predict(x)
    let loss = this.loss_layer.forward(score, t)
    return loss
}
TwoLayerNet.prototype.backward = function(dout) {
    dout = this.loss_layer.backward(dout)
    for(let i = this.layers.length-1; i > -1; i--) {
        dout = this.layers[i].backward(dout)
    }
    return dout
}

function Trainer(model, optimizer) {
    this.model = model
    this.optimizer = optimizer
    this.loss_list = []
    this.current_epoch = 0
}
Trainer.prototype.fit = function(x, t, max_epoch, batch_size, eval_interval) {
    let data_size = x.length
    let max_iters = Math.floor(data_size / batch_size)
    let model = this.model
    let optimizer = this.optimizer
    let total_loss = 0
    let loss_count = 0

    start_time = Date.now()
    for(let epoch = 0; epoch < max_epoch; epoch++) {
        for(let iters = 0; iters < max_iters; iters++) {
            let batch_x = x.slice(iters*batch_size, (iters+1)*batch_size)
            let batch_t = t.slice(iters*batch_size, (iters+1)*batch_size)
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1
        }
        if(eval_interval !== undefined && (epoch % eval_interval == 0)) {
            avg_loss = total_loss / loss_count
            elapsed_time = (Date.now() - start_time)/1000
            console.log("에폭 " + (this.current_epoch+1) + " / " + max_epoch +
            " | 시간 " + (elapsed_time.toFixed(2)) + "(s) | 손실 " + (avg_loss.toFixed(2)))
            this.loss_list.push(avg_loss)
            total_loss = 0
            loss_count = 0
        }
        this.current_epoch += 1
    }
}
Trainer.prototype.getLossList = function() {
    return this.loss_list
}


let max_epoch = 1
let batch_size = 30
let hidden_size = 10
let learning_rate = 0.1

let data = load_data()
let x = data[0]
let t = data[1]
let model = new TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
let optimizer = new Adam(lr=learning_rate)

let trainer = new Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, 1)