import numpy as np

class A:
    def __init__(self):
        self.x = 1

    def __call__(self):
        print(self.x)

a = A()
a()