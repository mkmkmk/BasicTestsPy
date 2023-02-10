# https://qiskit.org/textbook/preface.html
#
# pip install qiskit
from qiskit import QuantumCircuit, Aer, assemble
from qiskit import *
from math import pi
import numpy as np
from qiskit.visualization import plot_histogram, plot_bloch_multivector

sim = Aer.get_backend('qasm_simulator')

qc = QuantumCircuit(3)
for qubit in range(3):
    qc.h(qubit)

qc.draw()
svsim = Aer.get_backend('statevector_simulator')
qobj = assemble(qc)
final_state = svsim.run(qobj).result().get_statevector()

# pip install git+https://github.com/qiskit-community/qiskit-textbook.git#subdirectory=qiskit-textbook-src
from qiskit_textbook.tools import array_to_latex
array_to_latex(final_state, pretext="\\text{Statevector} = ")

#--------------
from qiskit import QuantumCircuit
qc = QuantumCircuit(2) # Create circuit with 2 qubits
qc.h(0)    # Do H-gate on q0
qc.cx(0,1) # Do CNOT on q1 controlled by q0
qc.measure_all()
qc.draw()

#--------------
n = 8
qc_output = QuantumCircuit(n, n)
qc_output.draw()
for j in range(n):
    qc_output.measure(j,j)
qc_output.draw()

#--------------
qc = QuantumCircuit(2)
qc.cx(0,1) # NOT
qc.draw()
#--------------
qc_encode = QuantumCircuit(n)
qc_encode.x(7)
qc_encode.draw()

#--------------
qc = QuantumCircuit(2,2)
qc.x(0)
#qc.x(1)
qc.cx(0,1)
qc.measure(0,0)
qc.measure(1,1)
qc.draw()
counts = sim.run(assemble(qc)).result().get_counts()
plot_histogram(counts)

# -----------------------
# half adder
# https://qiskit.org/textbook/ch-states/atoms-computation.html
# -----------------------
qc_ha = QuantumCircuit(4,2)
# encode inputs in qubits 0 and 1
qc_ha.x(0) # For a=0, remove the this line. For a=1, leave it.
#qc_ha.x(1) # For b=0, remove the this line. For b=1, leave it.
qc_ha.barrier()
# use cnots to write the XOR of the inputs on qubit 2
qc_ha.cx(0,2)
qc_ha.cx(1,2)
# use ccx to write the AND of the inputs on qubit 3
qc_ha.ccx(0,1,3)
qc_ha.barrier()
# extract outputs
qc_ha.measure(2,0) # extract XOR value
qc_ha.measure(3,1) # extract AND value

qc_ha.draw()
counts = sim.run(assemble(qc_ha)).result().get_counts()
plot_histogram(counts)

# -----------------------
qc = QuantumCircuit(2)
#qc.x(0)
qc.h(0)
qc.cx(0,1)
#qc.ry(pi/8, 0)
qc.draw()
svsim = Aer.get_backend('statevector_simulator')
qobj = assemble(qc)
state = svsim.run(qobj).result().get_statevector()
print(state)
plot_bloch_multivector(state)


# -----------------------
from qiskit.circuit.library.standard_gates.ry import RYGate
ry1 = RYGate(0)
ry1.to_matrix()
ry1 = RYGate(2*pi/4)
ry1.to_matrix()
ry1 = RYGate(2*pi/2/20)
ry1.to_matrix()


# -----------------------

# -----------------------
N = 10
qc = QuantumCircuit(1, 1)
qc.x(0)
qc.h(0)
#qc.ry(2*pi/2/N, 0) # *2 <= Bloch sphere theta
#qc.cx(0, 1)
#qc.measure(1, 1)

#qc.barrier()
#qc.measure(0, 0)
qc.draw()
#sim = Aer.get_backend('qasm_simulator')
svsim = Aer.get_backend('statevector_simulator')
result = svsim.run(assemble(qc)).result()
result.get_statevector()
# -----------------------
# -----------------------
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0,1)
qc.h(0)
qc.h(1)
qc.mct(0,1)
qc.draw()
# -----------------------
# -----------------------
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0,1)
#qc.h(0)
qc.ry(2*pi/8, 1)
qc.measure(0,0)
qc.measure(1,1)

qc.draw()
counts = sim.run(assemble(qc)).result().get_counts()
plot_histogram(counts)

# -----------------------
from qiskit_textbook.widgets import dj_widget
dj_widget(size="small", case="balanced")
input()

# -----------------------
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
# -----------------------
# -----------------------
# https://qiskit.org/textbook/ch-algorithms/bernstein-vazirani.html
n=3
bv_circuit = QuantumCircuit(n+1, n)
bv_circuit.x(n)
bv_circuit.h(n)
#bv_circuit.z(n)
for i in range(n):
    bv_circuit.h(i)
bv_circuit.barrier()
s = '101'
s = s[::-1] # reverse s to fit qiskit's qubit ordering
for q in range(n):
    if s[q] == '1':
        bv_circuit.cx(q, n)
    #else:
    #    bv_circuit.i(q)
bv_circuit.barrier()
for i in range(n):
    bv_circuit.h(i)
for i in range(n):
    bv_circuit.measure(i, i)
bv_circuit.draw()
svsim = Aer.get_backend('statevector_simulator')
result = svsim.run(assemble(bv_circuit)).result()
result.get_statevector()

sim = Aer.get_backend('qasm_simulator')
counts = sim.run(assemble(bv_circuit)).result().get_counts()
plot_histogram(counts)

# -----------------------
qc = QuantumCircuit(4)
qc.swap(2,3)
qc.swap(1,2)
qc.swap(0,1)
qc.draw()

# -----------------------
# -----------------------
# -----------------------
# -----------------------
