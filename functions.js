(function() {
    "use strict"
    const Arr = require("./Arr")
    const {Operation, Variable, List, Config, sum_to, broadcast_to, as_variable, as_array} = require("./core")
    const utils = require("./utils")

    function Sin() {
        return Operation.inherit(Sin)
    }
    
    Sin.prototype.__proto__ = Operation.prototype
    
    Sin.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.sin(v))
        return y
    }
    
    Sin.prototype.backward = function(gy) {
        let x = this.inputs[0]
        let gx = cos(x).mul(gy)
        return gx
    }


    function Cos() {
        return Operation.inherit(Cos)
    }

    Cos.prototype.__proto__ = Operation.prototype

    Cos.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.cos(v))
        return y
    }

    Cos.prototype.backward = function(gy) {
        let x = this.inputs[0]
        let gx = sin(x).mul(-1).mul(gy)
        return gx
    }


    function Tanh() {
        return Operation.inherit(Tanh)
    }

    Tanh.prototype.__proto__ = Operation.prototype

    Tanh.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.tanh(v))
        return y
    }

    Tanh.prototype.backward = function(gy) {
        let y = this.outputs[0]
        let gx = gy.mul(y.pow(2).rminus(1))
        return gx
    }


    function Exp() {
        return Operation.inherit(Exp)
    }

    Exp.prototype.__proto__ = Operation.prototype

    Exp.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.exp(v))
        return y
    }

    Exp.prototype.backward = function(gy) {
        let y = this.outputs[0]
        let gx = gy.mul(y)
        return gx
    }


    function Log() {
        return Operation.inherit(Log)
    }

    Log.prototype.__proto__ = Operation.prototype

    Log.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.log(v))
        return y
    }

    Log.prototype.backward = function(gy) {
        let x = this.inputs[0]
        let gx = gy.div(x)
        return gx
    }


    function MatMul() {
        return Operation.inherit(MatMul)
    }

    MatMul.prototype.__proto__ = Operation.prototype

    MatMul.prototype.forward = function(x, W) {
        let y = x.matmul(W)
        return y
    }

    MatMul.prototype.backward = function(gy) {
        let x = this.inputs[0]
        let W = this.inputs[1]
        let gx = matmul(gy, W.T)
        let gW = matmul(x.T, gy)
        return List(gx, gW)
    }


    function Linear() {
        return Operation.inherit(Linear)
    }

    Linear.prototype.__proto__ = Operation.prototype

    Linear.prototype.forward = function(x, W, b) {
        let y = x.matmul(W)
        if(b != null) {
            y = y.plus(b)
        }
        return y
    }

    Linear.prototype.backward = function(gy) {
        let [x, W, b] = this.inputs
        let gb = b.data != null ? sum_to(gy, b.shape) : null
        let gx = matmul(gy, W.T)
        let gW = matmul(x.T, gy)
        return List(gx, gW, gb)
    }


    function Im2col(kernel_size, stride, pad, to_matrix) {
        let result = Operation.inherit(Im2col)
        result.input_shape = null
        result.kernel_size = kernel_size
        result.stride = stride
        result.pad = pad
        result.to_matrix = to_matrix
        return result
    }

    Im2col.prototype.__proto__ = Operation.prototype

    Im2col.prototype.forward = function(x) {
        this.input_shape = x.shape
        let y = im2col_array(x, this.kernel_size, this.stride, this.pad, this.to_matrix)
        return y
    }

    Im2col.prototype.backward = function(gy) {
        let gx = col2im(gy, this.input_shape, this.kernel_size, this.stride, this.pad, this.to_matrix)
        return gx
    }


    function Col2im(input_shape, kernel_size, stride, pad, to_matrix) {
        let result = Operation.inherit(Col2im)
        result.input_shape = input_shape
        result.kernel_size = kernel_size
        result.stride = stride
        result.pad = pad
        result.to_matrix = to_matrix
        return result
    }

    Col2im.prototype.__proto__ = Operation.prototype

    Col2im.prototype.forward = function(x) {
        let y = col2im_array(x, this.input_shape, this.kernel_size, this.stride, this.pad, this.to_matrix)
        return y
    }

    Col2im.prototype.backward = function(gy) {
        let gx = im2col(gy, this.kernel_size, this.stride, this.pad, this.to_matrix)
        return gx
    }


    function Pooling(kernel_size, stride, pad) {
        stride = stride === undefined ? 1 : stride
        pad = pad === undefined ? 0 : pad

        let result = Operation.inherit(Pooling)
        result.kernel_size = kernel_size
        result.stride = stride
        result.pad = pad
        return result
    }

    Pooling.prototype.__proto__ = Operation.prototype

    Pooling.prototype.forward = function(x) {
        let col = im2col_array(x, this.kernel_size, this.stride, this.pad, false)
        let [N, C, KH, KW, OH, OW] = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        let y = col.max(2)
        return y
    }


    function AveragePooling(kernel_size, stride, pad) {
        stride = stride === undefined ? 1 : stride
        pad = pad === undefined ? 0 : pad

        let result = Operation.inherit(AveragePooling)
        result.kernel_size = kernel_size
        result.stride = stride
        result.pad = pad
        result.input_shape = null
        return result
    }

    AveragePooling.prototype.__proto__ = Operation.prototype

    AveragePooling.prototype.forward = function(x) {
        this.input_shape = x.shape
        let col = im2col_array(x, this.kernel_size, this.stride, this.pad, false)
        let y = col.mean([2, 3])
        return y
    }

    AveragePooling.prototype.backward = function(gy) {
        let [N, C, OH, OW] = gy.shape
        let [KW, KH] = utils.pair(this.kernel_size)

        gy = gy.div(KW * KH)
        let gcol = broadcast_to(gy.reshape(-1), [KH, KW, N*C*OH*OW])
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)

        let gx = col2im(gcol, this.input_shape, this.kernel_size, this.stride, this.pad, false)
        return gx
    }


    function Sigmoid() {
        return Operation.inherit(Sigmoid)
    }

    Sigmoid.prototype.__proto__ = Operation.prototype

    Sigmoid.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.tanh(v * 0.5) * 0.5 + 0.5)
        return y
    }

    Sigmoid.prototype.backward = function(gy) {
        let y = this.outputs[0]
        let gx = gy.mul(y).mul(y.rminus(1))
        return gx
    }


    function ReLU() {
        return Operation.inherit(ReLU)
    }

    ReLU.prototype.__proto__ = Operation.prototype

    ReLU.prototype.forward = function(x) {
        let y = x.deepMap(v => Math.max(v, 0))
        return y
    }

    ReLU.prototype.backward = function(gy) {
        let x = this.inputs[0]
        let mask = x.deepMap(v => Number(v > 0))
        let gx = gy.mul(mask)
        return gx
    }


    function Softmax(axis) {
        let result = Operation.inherit(Softmax)
        result.axis = axis === undefined ? 1 : axis
        return result
    }

    Softmax.prototype.__proto__ = Operation.prototype

    Softmax.prototype.forward = function(x) {
        let y = x.minus(x.max(this.axis, true))
        y = y.deepMap(v => Math.exp(v))
        y = y.div(y.sum(this.axis, true))
        return y
    }

    Softmax.prototype.backward = function(gy) {
        let y = this.outputs[0]
        let gx = y.mul(gy)
        let sumdx = gx.sum(this.axis, true)
        gx = gx.minus(y.mul(sumdx))
        return gx
    }


    function MeanSquaredError() {
        return Operation.inherit(MeanSquaredError)
    }

    MeanSquaredError.prototype.__proto__ = Operation.prototype

    MeanSquaredError.prototype.forward = function(x0, x1) {
        let diff = x0.minus(x1)
        let y = diff.pow(2).sum() / diff.length
        return y
    }

    MeanSquaredError.prototype.backward = function(gy) {
        let x0 = this.inputs[0]
        let x1 = this.inputs[1]
        let diff = x0.minus(x1)
        let gx0 = gy.mul(diff).mul(2 / diff.length)
        let gx1 = gx0.mul(-1)
        return List(gx0, gx1)
    }


    function SoftmaxCrossEntropy() {
        return Operation.inherit(SoftmaxCrossEntropy)
    }

    SoftmaxCrossEntropy.prototype.__proto__ = Operation.prototype

    SoftmaxCrossEntropy.prototype.forward = function(x, t) {
        let N = x.shape[0]
        let log_z = utils.logsumexp(x, 1)
        let log_p = x.minus(log_z)
        t = t.flat()
        let log_p2 = 0
        for(let i = 0; i < N; i++) {
            log_p2 += log_p[i][t[i]]
        }
        let y = -log_p2 / N
        return y
    }

    SoftmaxCrossEntropy.prototype.backward = function(gy) {
        let [x, t] = this.inputs
        let [N, CLS_NUM] = x.shape

        gy = gy.mul(1/N)
        let y = softmax(x)
        let t_onehot = Arr()
        for(let i = 0; i < t.length; i++) {
            let arr = Arr.zeros(CLS_NUM)
            arr[t.data[i]] = 1
            t_onehot.push(arr)
        }
        y = y.minus(t_onehot).mul(gy)
        return y
    }


    function sin(x) {
        return Sin()(x)
    }

    function cos(x) {
        return Cos()(x)
    }

    function tanh(x) {
        return Tanh()(x)
    }

    function exp(x) {
        return Exp()(x)
    }

    function log(x) {
        return Log()(x)
    }

    function matmul(x, W) {
        return MatMul()(x, W)
    }

    function linear(x, W, b) {
        b = b === undefined ? null : b
        return Linear()(x, W, b)
    }

    function im2col(x, kernel_size, stride, pad, to_matrix) {
        stride = stride === undefined ? 1 : stride
        pad = pad === undefined ? 0 : pad
        to_matrix = to_matrix === undefined ? true : to_matrix
        return Im2col(kernel_size, stride, pad, to_matrix)(x)
    }

    function col2im(x, input_shape, kernel_size, stride, pad, to_matrix) {
        stride = stride === undefined ? 1 : stride
        pad = pad === undefined ? 0 : pad
        to_matrix = to_matrix === undefined ? true : to_matrix
        return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)
    }

    function average_pooling(x, kernel_size, stride, pad) {
        return AveragePooling(kernel_size, stride, pad)(x)
    }

    function sigmoid(x) {
        return Sigmoid()(x)
    }

    function relu(x) {
        return ReLU()(x)
    }

    function softmax(x, axis) {
        return Softmax(axis)(x)
    }

    function mean_squared_error(x0, x1) {
        return MeanSquaredError()(x0, x1)
    }

    function softmax_cross_entropy(x, t) {
        return SoftmaxCrossEntropy()(x, t)
    }

    function flatten(x) {
        return x.reshape(x.shape[0], -1)
    }

    
    function accuracy(y, t) {
        y = as_variable(y)
        t = as_variable(t)

        let pred = y.data.argmax(1).reshape(t.shape)
        let result = pred.cal(t.data, (a, b) => a === b)
        let acc = result.sum().div(result.length)
        return as_variable(as_array(acc))
    }

    function dropout(x, dropout_ratio) {
        dropout_ratio = dropout_ratio === undefined ? 0.5 : dropout_ratio
        x = as_variable(x)

        if(Config.train) {
            let scale = 1 - dropout_ratio
            let y = x.deepMap(v => (Math.random() > dropout_ratio) * v / scale)
            return y
        } else {
            return x
        }
    }

    function conv2d_simple(x, W, b, stride, pad) {
        b = b === undefined ? null : b
        stride = stride === undefined ? 1 : stride
        pad = pad === undefined ? 0 : pad
        x = as_variable(x)
        let Weight = as_variable(W)

        let [N, _, H, a] = x.shape
        W = a
        let [OC, C, KH, KW] = Weight.shape
        let [SH, SW] = utils.pair(stride)
        let [PH, PW] = utils.pair(pad)
        let OH = utils.get_conv_outsize(H, KH, SH, PH)
        let OW = utils.get_conv_outsize(W, KW, SW, PW)

        let col = im2col(x, [KH, KW], stride, pad, true)
        Weight = Weight.reshape(OC, -1).transpose()
        let t = linear(col, Weight, b)
        let y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
        return y
    }

    function pooling_simple(x, kernel_size, stride, pad) {
        stride = stride === undefined ? 1 : stride
        pad = pad === undefined ? 0 : pad
        x = as_variable(x)

        let [N, C, H, W] = x.shape
        let [KH, KW] = utils.pair(kernel_size)
        let [SH, SW] = utils.pair(stride)
        let [PH, PW] = utils.pair(pad)
        let OH = utils.get_conv_outsize(H, KH, SH, PH)
        let OW = utils.get_conv_outsize(W, KW, SW, PW)

        let col = im2col(x, kernel_size, stride, pad, true)
        col = col.reshape(-1, KH * KW)
        let y = col.max(1)
        y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
        return y
    }

    function im2col_array(img, kernel_size, stride, pad, to_matrix) {
        to_matrix = to_matrix === undefined ? true : to_matrix
        let [N, C, H, W] = img.shape
        let [KH, KW] = utils.pair(kernel_size)
        let [SH, SW] = utils.pair(stride)
        let [PH, PW] = utils.pair(pad)
        let OH = utils.get_conv_outsize(H, KH, SH, PH)
        let OW = utils.get_conv_outsize(W, KW, SW, PW)

        let img2 = Arr.zeros(N, C)
        for(let i = 0; i < N; i++) {
            for(let j = 0; j < C; j++) {
                let temp = Arr.zeros(H + PH*2 + SH - 1, W + PW*2 + SW - 1)
                temp.overlap(img[i][j], [PH, PW], [H+PH, W+PW])
                img2[i][j] = temp
            }
        }
        img = null
        let col = Arr.zeros(N, C, KH, KW, OH, OW)

        for(let j = 0; j < KH; j++) {
            let j_lim = j + SH * OH
            for(let i = 0; i < KW; i++) {
                let i_lim = i + SH * OW
                for(let k = 0; k < N; k++) {
                    for(let l = 0; l < C; l++) {
                        let temp = img2[k][l].slice([j, i], [j_lim, i_lim])
                        temp = temp.filter((_, i) => i%SH === 0)
                        col[k][l][j][i] = temp.map(v => v.filter((_, i) => i%SW === 0))
                    }
                }
            }
        }
        img2 = null

        if(to_matrix) {
            col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
        }
        return col
    }

    function col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix) {
        to_matrix = to_matrix === undefined ? true : to_matrix
        let [N, C, H, W] = img_shape
        let [KH, KW] = utils.pair(kernel_size)
        let [SH, SW] = utils.pair(stride)
        let [PH, PW] = utils.pair(pad)
        let OH = utils.get_conv_outsize(H, KH, SH, PH)
        let OW = utils.get_conv_outsize(W, KW, SW, PW)

        if(to_matrix) {
            col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
        }

        let img = Arr.zeros(N, C, H + PH*2 + SH - 1, W + PW*2 + SW - 1)
        for(let j = 0; j < KH; j++) {
            for(let i = 0; i < KW; i++) {
                for(let k = 0; k < N; k++) {
                    for(let l = 0; l < C; l++) {
                        for(let m = 0; m < OH; m++) {
                            for(let n = 0; n < OW; n++) {
                                img[k][l][j+m*SH][i+n*SW] += col[k][l][j][i][m][n]
                            }
                        }
                    }
                }
            }
        }
        return img.slice([0, 0, PH, PW], [N, C, H + PH, W + PW])
    }


    module.exports = {
        sin : sin,
        cos : cos,
        tanh : tanh,
        exp : exp,
        log : log,
        matmul : matmul,
        linear : linear,
        sigmoid : sigmoid,
        relu : relu,
        softmax : softmax,
        mean_squared_error : mean_squared_error,
        softmax_cross_entropy : softmax_cross_entropy,
        flatten : flatten,
        accuracy : accuracy,
        dropout : dropout,
        im2col : im2col,
        conv2d_simple : conv2d_simple,
        pooling : pooling_simple,
        average_pooling : average_pooling
    }
})()