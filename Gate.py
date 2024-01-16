from Quantum_Helper import Quantum_Helper as qh
import numpy as np

class Gate:

    gates = {}

    def __init__(self, gate):
        # Assert shape
        shape = np.shape(gate)
        assert len(shape) == 2
        assert shape[0] == shape[1]
        assert shape[0] > 1

        # Assert to be unitary
        assert Gate.__is_valid(gate) == True

        # Assert dimension
        n_qubits = np.log2(len(gate))
        assert n_qubits > 0
        assert round(n_qubits) == n_qubits

        # Save info to object
        self.__n_qubits = n_qubits
        self.__gate = gate

    @staticmethod
    def create_by_composing(gates):
        composite_gate = gates[0]
        for gate in gates[1:]:
            composite_gate = np.dot(composite_gate, gate)

        return Gate(composite_gate)

    @staticmethod
    def __is_valid(gate):
        return np.allclose(np.dot(gate, qh.compute_adjoint(gate)), np.identity(len(gate)))
        
    # cteja TODO: delete
    def get_gate(self):
        return self.__gate

    @staticmethod
    def compute_standard_gates():
        Gate.gates["H"] = Gate(np.array([[1, 1], [1, -1]]) / qh.sqrt2)
        Gate.gates["X"] = Gate(np.array([[0, 1], [1, 0]]))
        Gate.gates["S"] = Gate(np.array([[1, 0], [0, 1j]]))
        Gate.gates["CNOT"] = Gate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
        #Gate.gates["CH"] = Gate(np.array([[1, 0, 0, 0], [0, qh.sqrt2, 0, qh.sqrt2], [0, 0, 1, 0], [0, qh.sqrt2, 0, qh.sqrt2]])) #TODO
        Gate.gates["SWAP"] = Gate(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0 ,0], [0, 0, 0, 1]]))
        Gate.gates["CNOT10"] = Gate(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1 ,0], [0, 0, 0, 1]]))
        Gate.gates["TOFFOLI"] = Gate(np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 1, 0, 0, 0, 0, 0], [0 ,0 ,0 ,1 ,0 ,0 ,0 ,0], [0 ,0 ,0 ,0 ,1 ,0 ,0 ,0], [0 ,0 ,0 ,0 ,0 ,1 ,0 ,0], [0 ,0 ,0 ,0 ,0 ,0 ,0 ,1], [0 ,0 ,0 ,0 ,0 ,0 ,1 ,0]]))
        Gate.gates["PAULI_Y"] = Gate(np.array([[0, -1j], [1j, 0]]))



# cteja TODO: move in quantum simulator file
Gate.compute_standard_gates()
