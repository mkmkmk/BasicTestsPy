# https://en.wikipedia.org/wiki/Automatic_differentiation#Python
# - niedługo pewnie jakiś idiota usunie też ten kod z wikipedii

class Expression:
    def __add__(self, other):
        return Plus(self, other)
    def __mul__(self, other):
        return Multiply(self, other)

class Variable(Expression):
    def __init__(self, value):
        self.value = value
        self.partial = 0

    def evaluate(self):
        pass

    def derive(self, seed):
        self.partial += seed

class Plus(Expression):
    def __init__(self, expressionA, expressionB):
        self.expressionA = expressionA
        self.expressionB = expressionB
        self.value = None

    def evaluate(self):
        self.expressionA.evaluate()
        self.expressionB.evaluate()
        self.value = self.expressionA.value + self.expressionB.value

    def derive(self, seed):
        self.expressionA.derive(seed)
        self.expressionB.derive(seed)

class Multiply(Expression):
    def __init__(self, expressionA, expressionB):
        self.expressionA = expressionA
        self.expressionB = expressionB
        self.value = None

    def evaluate(self):
        self.expressionA.evaluate()
        self.expressionB.evaluate()
        self.value = self.expressionA.value * self.expressionB.value

    def derive(self, seed):
        self.expressionA.derive(self.expressionB.value * seed)
        self.expressionB.derive(self.expressionA.value * seed)

# Example: Finding the partials of z = x * (x + y) + y * y at (x, y) = (2, 3)
x = Variable(2)
y = Variable(3)
z = x * (x + y) + y * y
z.evaluate()
print("z =", z.value)        # Output: z = 19
z.derive(1)
print("∂z/∂x =", x.partial)  # Output: ∂z/∂x = 7
print("∂z/∂y =", y.partial)  # Output: ∂z/∂y = 8