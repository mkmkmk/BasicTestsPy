from qiskit import QuantumCircuit, Aer, assemble
from qiskit import *
from math import pi
import numpy as np
from qiskit.visualization import plot_histogram, plot_bloch_multivector

def c_amod15(a, power):
    """Controlled multiplication by a mod 15"""
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11 or 13")
    U = QuantumCircuit(4)
    for iteration in range(power):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if a == 11:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U

qc = QuantumCircuit(5, 4)
qc.x(0)
qc.x(4)
qc.append(c_amod15(7, 2), [i for i in range(5)])
qc.measure([1+i for i in range(4)], [i for i in range(4)])
qc.draw()

qasm_sim = Aer.get_backend('qasm_simulator')
t_qc = transpile(qc, qasm_sim)
qobj = assemble(t_qc)
results = qasm_sim.run(qobj).result()
counts = results.get_counts()
plot_histogram(counts)


def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(pi/2**(n-qubit), qubit, n)
    # At the end of our function, we call the same function again on
    # the next qubits (we reduced n by one earlier in the function)
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(n):
    """QFT on the first n qubits in circuit"""
    qc = QuantumCircuit(n)
    qft_rotations(qc, n)
    swap_registers(qc, n)
    qc.name = "QFT"
    return qc

n_count = 8
n_count = 4
a = 11
a = 7

# --------------------
qc = QuantumCircuit(n_count + 4, n_count)
for q in range(n_count):
    qc.h(q)
qc.x(3+n_count)

for q in range(n_count):
    qc.append(c_amod15(a, 2**q),
             [q] + [i+n_count for i in range(4)])

#qc.append(qft_dagger(n_count), range(n_count))
qc.append(qft(n_count), range(n_count))

qc.measure(range(n_count), range(n_count))
qc.draw(fold=-1)

qasm_sim = Aer.get_backend('qasm_simulator')
t_qc = transpile(qc, qasm_sim)
qobj = assemble(t_qc)
results = qasm_sim.run(qobj).result()
counts = results.get_counts()
plot_histogram(counts)

svsim = Aer.get_backend('statevector_simulator')
t_qc = transpile(qc, qasm_sim)
result = svsim.run(assemble(t_qc)).result()
result.get_statevector()

# --------------------
c1 = ClassicalRegister(n_count, name = "c1")
c2 = ClassicalRegister(4, name = "c2")
q1 = QuantumRegister(n_count + 4, name = "q1")
q2 = QuantumRegister(n_count + 4, name = "q2")
qc = QuantumCircuit(qr, c1, c2)
for q in range(n_count):
    qc.h(q)
qc.x(3+n_count)

for q in range(n_count):
    qc.append(c_amod15(a, 2**q),
             [q] + [i+n_count for i in range(4)])

#qc.append(qft_dagger(n_count), range(n_count))
#qc.barrier()
qc.measure([i+n_count for i in range(4)], [i+n_count for i in range(4)])
qc.append(qft(n_count), range(n_count))
qc.barrier()
qc.measure(range(n_count), range(n_count))
#qc.measure(range(n_count), c1)

qc.draw(fold=-1)

qasm_sim = Aer.get_backend('qasm_simulator')
t_qc = transpile(qc, qasm_sim)
qobj = assemble(t_qc)
results = qasm_sim.run(qobj).result()
counts = results.get_counts()
plot_histogram(counts)
