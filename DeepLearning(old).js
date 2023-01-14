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
    return arr1.map((row) => arr2[0].map((x,y) => row.reduce((a,b,c) => a + b * arr2[c][y], 0)))
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

function softmax(x) {
    if(x.ndim() === 2) {
        let max = x.map(v => Math.max.apply(null, v))
        max = max.reshape(max.length, 1)
        x = x.minus(max)
        x = x.deepMap(v => Math.exp(v))
        let sum = x.map(v => v.sum())
        x = x.cal(sum.reshape(sum.length, 1), (a,b) => a/b)
    } else if(x.ndim() === 1) {
        x = x.minus(Math.max.apply(null, x))
        let exp = x.deepMap(v => Math.exp(v))
        x = exp.cal(exp.sum(), (a,b) => a/b)
    }
    return x
}

function cross_entropy_error(y, t){
    if(y.ndim() === 1) {
        t = t.reshape(1, t.size())
        y = y.reshape(1, y.size())
    }
    // 정답 데이터가 원핫 벡터일 경우 정답 레이블 인덱스로 변환
    if(t.size() === y.size()) {
        t = t.map(v => v.indexOf(Math.max.apply(null, v)))
    }
    batch_size = y.shape()[0]
    let temp = 0
    for(let i = 0; i < batch_size; i++) {
        temp += Math.log(y.get([i, t[i]]) + 1e-7)
    }
    return -temp / batch_size
}

function load_data() {
    let N = 300  //클래스당 샘플 수
    let DIM = 2  //데이터 요소 수
    let CLS_NUM = 3  //클래스 수

    let x = Array.zeros(N*CLS_NUM, DIM)
    let t = Array.zeros(N*CLS_NUM, CLS_NUM)

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

const DeepLearning = {
    "layers" : {},
    "optimizer" : {},
    "trainer" : {},
    "util" : {}
}

function Sigmoid() {
    this.params = []
    this.grads = []
    this.out = null
}
Sigmoid.prototype.forward = function(x) {
    let out = x.deepMap(v => 1 / (1 + Math.exp(-v)))
    this.out = out
    return out
}
Sigmoid.prototype.backward = function(dout) {
    let dx = dout.cal(this.out, (a,b) => a * (1 - b) * b)
    return dx
}

function Relu() {
    this.params = []
    this.grads = []
    this.out = null
}
Relu.prototype.forward = function(x) {
    let out = x.deepMap(v => v > 0 ? v : 0)
    this.out = out
    return out
}
Relu.prototype.backward = function(dout) {
    let dx = dout.cal(this.out, (a,b) => b > 0 ? a : 0)
    return dx
}

function SoftmaxWithLoss() {
    this.params = []
    this.grads = []
    this.y = []
    this.t = []
}
SoftmaxWithLoss.prototype.forward = function(x, t) {
    this.t = t
    this.y = softmax(x)

    //정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
    if(this.t.size() === this.y.size()) {
        this.t = this.t.map(v => v.indexOf(Math.max.apply(null, v)))
    }

    loss = cross_entropy_error(this.y, this.t)
    return loss
}
SoftmaxWithLoss.prototype.backward = function(dout) {
    dout = dout === undefined ? 1 : dout
    batch_size = this.t.shape()[0]

    dx = copy(this.y)
    for(let i = 0; i < batch_size; i++) {
        dx.put([i, this.t[i]], dx.get([i, this.t[i]])-1 )
    }
    dx = dx.cal(dout, (a,b) => a*b / batch_size)
    return dx
}

function Affine(W, b) {
    this.params = [W, b]
    this.grads = [Array.zeros(W.shape()), Array.zeros(b.shape())]
    this.x = null
}
Affine.prototype.forward = function(x) {
    let W = this.params[0]
    let b = this.params[1]
    out = matmul(x, W).plus(b)
    this.x = x
    return out
}
Affine.prototype.backward = function(dout) {
    let W = this.params[0]
    let dx = matmul(dout, W.T())
    let dW = matmul(this.x.T(), dout)
    let db = dout.sum(0)

    this.grads[0] = copy(dW)
    this.grads[1] = copy(db)
    return dx
}

function SGD(lr) {
    this.lr = lr === undefined ? 0.01 : lr
}
SGD.prototype.update = function(params, grads) {
    for(let i = 0; i < params.length; i++) {
        Array.For(params[i], index => params[i].put(index, params[i].get(index) - grads[i].get(index)*this.lr))
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

function TwoLayerNet(input_size, hidden_size, output_size) {
    let I = input_size, 
        H = hidden_size, 
        O = output_size

    let W1 = Array.zeros(I, H).deepMap(v => gaussRandom()*0.01)
    let b1 = Array.zeros(H)
    let W2 = Array.zeros(H, O).deepMap(v => gaussRandom()*0.01)
    let b2 = Array.zeros(O)

    this.layers = [
        new Affine(W1, b1),
        new Sigmoid(),
        new Affine(W2, b2)
    ]
    this.loss_layer = new SoftmaxWithLoss()

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

            if(eval_interval !== undefined && (iters % eval_interval === 0)) {
                avg_loss = total_loss / loss_count
                elapsed_time = (Date.now() - start_time)/1000
                console.log("에폭 " + (this.current_epoch+1) + " | 반복 " + (iters+1) + " / " + max_iters +
                " | 시간 " + (elapsed_time.toFixed(2)) + "(s) | 손실 " + (avg_loss.toFixed(2)))
                this.loss_list.push(avg_loss)
                total_loss = 0
                loss_count = 0
            }
        }
        this.current_epoch += 1
    }
}
Trainer.prototype.getLossList = function() {
    return this.loss_list
}


let max_epoch = 3
let batch_size = 30
let hidden_size = 10
let learning_rate = 0.5

let data = load_data()
let x = data[0]
let t = data[1]
let model = new TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
let optimizer = new Adam(lr=learning_rate)

let trainer = new Trainer(model, optimizer)
trainer.fit(x, t, max_epoch, batch_size, 10)
console.log(trainer.getLossList())