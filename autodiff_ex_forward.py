# https://en.wikipedia.org/wiki/Automatic_differentiation
# - usunięty przykład z wikipedii!!

class ValueAndPartial:
    def __init__(self, value, partial):
        self.value = value
        self.partial = partial

    def to_list(self):
        return [self.value, self.partial]

class Expression:
    def __add__(self, other):
        return Plus(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

class Variable(Expression):
    def __init__(self, value):
        self.value = value

    def evaluate_and_derive(self, variable):
        partial = 1 if self == variable else 0
        return ValueAndPartial(self.value, partial)

class Plus(Expression):
    def __init__(self, expression_a, expression_b):
        self.expression_a = expression_a
        self.expression_b = expression_b

    def evaluate_and_derive(self, variable):
        value_a, partial_a = self.expression_a.evaluate_and_derive(variable).to_list()
        value_b, partial_b = self.expression_b.evaluate_and_derive(variable).to_list()
        return ValueAndPartial(value_a + value_b, partial_a + partial_b)

class Multiply(Expression):
    def __init__(self, expression_a, expression_b):
        self.expression_a = expression_a
        self.expression_b = expression_b

    def evaluate_and_derive(self, variable):
        value_a, partial_a = self.expression_a.evaluate_and_derive(variable).to_list()
        value_b, partial_b = self.expression_b.evaluate_and_derive(variable).to_list()
        return ValueAndPartial(value_a * value_b, value_b * partial_a + value_a * partial_b)

# Example: Finding the partials of z = x * (x + y) + y * y at (x, y) = (2, 3)
x = Variable(2)
y = Variable(3)
z = x * (x + y) + y * y
x_partial = z.evaluate_and_derive(x).partial
y_partial = z.evaluate_and_derive(y).partial
print("∂z / ∂x =", x_partial)  # Output: ∂z / ∂x = 7
print("∂z / ∂y =", y_partial)  # Output: ∂z / ∂y = 8

print("---")

x = Variable(2)

z = x*x*x
# 3x^2 = 3*4
print(z.evaluate_and_derive(x).to_list())

z = x*x*x*x
# 4x^3 = 4*8
print(z.evaluate_and_derive(x).to_list())

z = x*x*x*x*x
# 5x^4 = 5*16
print(z.evaluate_and_derive(x).to_list())

