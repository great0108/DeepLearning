function numerical_diff(f, x, eps=1e-4) {
    let x0 = new Variable(x.data.minus(eps))
    let x1 = new Variable(x.data.plus(eps))
    let y0 = f(x0)
    let y1 = f(x1)
    return y1.data.minus(y0.data).div(2 * eps)
}

function factorial(a) {
    let result = 1
    for(let i = a; i > 1; i--) {
        result *= i
    }
    return result
}

function my_sin(x, threshold) {
    threshold = threshold === undefined ? 0.0001 : threshold
    let y = 0
    for(let i = 0; i < 100000; i++) {
        let c = Math.pow(-1, i) / factorial(2 * i + 1)
        let t = x.pow(2 * i + 1).mul(c)
        y = t.plus(y)
        if(Math.abs(t.data) < threshold) {
            break
        }
    }
    return y
}
