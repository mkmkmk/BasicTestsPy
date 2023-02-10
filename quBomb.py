"""
-----------------------
Quantum Bomb / Quantum Zeno Effect / Watched Pot Effect
-----------------------
https://arxiv.org/abs/1404.1507
https://arxiv.org/pdf/hep-th/9305002.pdf
https://www.scottaaronson.com/qclec.pdf
-----------------------
https://qiskit.org/textbook/preface.html
pip install qiskit
-----------------------
Created on: 31 mar 2021
    Author: M. Krej

"""
from qiskit import QuantumCircuit, Aer, assemble
from math import pi
from qiskit.visualization import plot_histogram
oo
# select bomb or dud :
bomb = True
bomb = False

N = 10
qc = QuantumCircuit(2, 2)

for i in range(N):
    qc.ry(2*pi/2/N, 0) # *2 <= Bloch sphere theta
    qc.cx(0, 1)
    if not bomb:
        qc.cx(0, 1)  # CNOT*CNOT == I
    qc.measure(1, 1)
    qc.barrier()

qc.measure(0, 0)
qc.draw()
sim = Aer.get_backend('qasm_simulator')
counts = sim.run(assemble(qc)).result().get_counts()
plot_histogram(counts)

print("done")
