# https://en.wikipedia.org/wiki/Automatic_differentiation

class Dual:
    def __init__(self, realPart, infinitesimalPart=0):
        self.realPart = realPart
        self.infinitesimalPart = infinitesimalPart

    def __add__(self, other):
        return Dual(
            self.realPart + other.realPart,
            self.infinitesimalPart + other.infinitesimalPart
        )

    def __mul__(self, other):
        return Dual(
            self.realPart * other.realPart,
            other.realPart * self.infinitesimalPart + self.realPart * other.infinitesimalPart
        )

# Example: Finding the partials of z = x * (x + y) + y * y at (x, y) = (2, 3)
def f(x, y):
    return x * (x + y) + y * y
x = Dual(2)
y = Dual(3)
epsilon = Dual(0, 1)
a = f(x + epsilon, y)
b = f(x, y + epsilon)
print("∂z/∂x =", a.infinitesimalPart)  # Output: ∂z/∂x = 7
print("∂z/∂y =", b.infinitesimalPart)  # Output: ∂z/∂y = 8