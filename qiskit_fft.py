from qiskit import QuantumCircuit, Aer, assemble
from qiskit import *
from math import pi
import numpy as np
from qiskit.visualization import plot_histogram, plot_bloch_multivector

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

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

qc = QuantumCircuit(2)
qft_rotations(qc,2)
qc.draw()

qc = QuantumCircuit(3)
# qc.cp(pi/2, 0, 1)
qc.cp(pi/4, 0, 2)
qc.cp(pi/2, 1, 2)
qc.h(2)

qc = QuantumCircuit(3)
qft(qc,2)
qc.draw()
qc = QuantumCircuit(3)
qft(qc,3)
qc.draw()
qc = QuantumCircuit(4)
qft(qc,4)
qc.draw()

backend = Aer.get_backend('unitary_simulator')
job = execute(qc, backend)
result = job.result()
print(result.get_unitary(qc, decimals=3))
