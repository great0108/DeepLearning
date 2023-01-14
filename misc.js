function numerical_diff(f, x, eps=1e-4) {
    let x0 = new Variable(x.data.minus(eps))
    let x1 = new Variable(x.data.plus(eps))
    let y0 = f(x0)
    let y1 = f(x1)
    return y1.data.minus(y0.data).div(2 * eps)
}