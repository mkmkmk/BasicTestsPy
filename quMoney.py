"""
-----------------------
     money attack
-----------------------
https://arxiv.org/abs/1404.1507
https://www.scottaaronson.com/qclec.pdf

-----------------------
Created on: 31 mar 2021
    Author: M. Krej

"""
from qiskit import QuantumCircuit, Aer, assemble
from math import pi
from qiskit.visualization import plot_histogram

N = 10

# --------------
# |0〉, |1〉
qc = QuantumCircuit(2, 2)
#qc.x(1) # select |0〉/ |1〉
qc.barrier()
for i in range(N):
    qc.ry(2*pi/2/N, 0) # *2 <= Bloch sphere theta
    qc.cx(0, 1)
    qc.measure(1, 1)
    qc.barrier()

qc.measure(0,0)
qc.draw()

sim = Aer.get_backend('qasm_simulator')
result = sim.run(assemble(qc)).result()
counts = result.get_counts()
plot_histogram(counts)

# --------------
# |+〉, |-〉
qc = QuantumCircuit(2, 2)
qc.x(1) # select |+〉/ |-〉
qc.h(1)
qc.barrier()
for i in range(N):
    qc.ry(2*pi/2/N, 0) # *2 <= Bloch sphere theta
    qc.cx(0, 1)
    qc.h(1)
    qc.measure(1, 1)
    qc.barrier()

qc.measure(0, 0)
qc.draw()

sim = Aer.get_backend('qasm_simulator')
result = sim.run(assemble(qc)).result()
counts = result.get_counts()
plot_histogram(counts)
