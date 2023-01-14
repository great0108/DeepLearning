function a() {
    return 1
}
a.a = 1
console.log(a())
console.log(a.a)