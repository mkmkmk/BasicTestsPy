#https://github.com/HIPS/autograd

from autograd import grad
import autograd.numpy as np

from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
import matplotlib.pyplot as plt

def tanh(x):                 # Define a function
        y = np.exp(-2.0 * x)
        return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)

grad_tanh(1.0)

(tanh(1.0001) - tanh(0.9999)) / 0.0002


x = np.linspace(-7, 7, 200)


plt.plot(
    x, tanh(x),
    x, egrad(tanh)(x),
    x, egrad(egrad(tanh))(x),
    x, egrad(egrad(egrad(tanh)))(x),
    x, egrad(egrad(egrad(egrad(tanh))))(x),
    x, egrad(egrad(egrad(egrad(egrad(tanh)))))(x)
    #x, egrad(egrad(egrad(egrad(egrad(egrad(tanh))))))(x)
    )
plt.show()


tanh(0)
tanh(1)
