# https://en.wikipedia.org/wiki/Automatic_differentiation#Beyond_forward_and_reverse_accumulation
# ⟨x,x'⟩ = x + x'ε

class Dual:
    def __init__(self, real, deriv=0):
        # y
        self.real = real
        # y', to nie jest infinitesimal, epsilon-a tu jawnie nie ma, jest w wyprowadzeniu działań
        self.deriv = deriv

    def __add__(self, other):
        return Dual(
            self.real + other.real,
            self.deriv + other.deriv
        )

    def __mul__(self, other):
        return Dual(
            self.real * other.real,
            other.real * self.deriv + self.real * other.deriv
        )

# Example: Finding the derivs of z = x * (x + y) + y * y at (x, y) = (2, 3)
def f(x, y):
    return x * (x + y) + y * y
x = Dual(2)
y = Dual(3)

# y + y'ε = 0 + 1ε = ε
epsilon = Dual(0, 1)

# f( ⟨x,1⟩ ) = ⟨ f(x), f'(x) ⟩
a = f(x + epsilon, y)
b = f(x, y + epsilon)
print("∂z/∂x =", a.deriv)  # Output: ∂z/∂x = 7
print("∂z/∂y =", b.deriv)  # Output: ∂z/∂y = 8

print("----")
def f(x):
    return x*x*x
x = Dual(2)
a = f(x + epsilon)
print(a.deriv)

print("----")
def f(x):
    return x*x*x*x
x = Dual(2)
a = f(x + epsilon)
print(a.deriv)

print("----")
def f(x):
    return x*x*x*x*x
x = Dual(2)
a = f(x + epsilon)
print(a.deriv)

print("----")
x = Dual(2, 1)
a = x*x*x*x*x
print(a.deriv)
