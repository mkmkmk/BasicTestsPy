import numpy as np
from qiskit import *
from qiskit import Aer

#Changing the simulator
backend = Aer.get_backend('unitary_simulator')

#The circuit without measurement
circ = QuantumCircuit(4)
#circ.cx(0,3)
#circ.cx(1,3)
#circ.cx(2,3)
circ.swap(0,3)
circ.draw()

job = execute(circ, backend)
result = job.result()
print(result.get_unitary(circ, decimals=3))
