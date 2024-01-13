import numpy as np
import scipy.linalg as linalg
import cmath

class State:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=np.complex64)
        self.state[0] = 1

    def initialize_state(self, qubit_values):
        assert len(qubit_values) == self.n_qubits
        self.state = np.zeros(2**len(qubit_values), dtype=np.complex64)
        sumOfCoefficients = 0
        for i in range(len(qubit_values)):
            sumOfCoefficients = sumOfCoefficients + qubit_values[i] * 2**(len(qubit_values) - i - 1)
        self.state[sumOfCoefficients] = 1
        
    
    def apply_gate(self, gate, n_gate, starting_qubit):
        
        applied_gate = np.identity(1)

        for i in range(starting_qubit):
            applied_gate = np.kron(applied_gate, np.identity(2))
        
        applied_gate = np.kron(applied_gate, gate)

        for i in range(starting_qubit + n_gate, self.n_qubits):
            applied_gate = np.kron(applied_gate, np.identity(2))

        self.state = np.dot(self.state, applied_gate)
       
    def compute_probabilities(self):
        probabilities = np.abs(self.state)**2

        return probabilities

    def apply_H_gate(self, target_qubit):
        
        self.apply_gate(np.array([[1, 1], [1, -1]])/np.sqrt(2), 1, target_qubit)
    
    def apply_X_gate(self, target_qubit):
        
        self.apply_gate(np.array([[0, 1], [1, 0]]), 1, target_qubit)
    
    def apply_S_gate(self, target_qubit):
        
        self.apply_gate(np.array([[1, 0], [0, 1j]]), 1, target_qubit)

    def apply_Y_gate(self, target_qubit):

        self.apply_gate(np.array([[0, -1j], [1j, 0]]), 1, target_qubit)

    def apply_T_gate(self, target_qubit):

        self.apply_gate(np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]]), 1, target_qubit)

    def apply_Z_gate(self, target_qubit):

        self.apply_gate(np.array([[1, 0], [0, -1]]), 1, target_qubit)

    def apply_CY_gate(self, target_qubit):

        self.apply_gate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0 ,0 ,0 ,1j], [0, 0, -1j, 0]]), 2, target_qubit)

    def apply_CZ_gate(self, target_qubit):

        self.apply_gate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0 ,0 ,1 ,0], [0, 0, 0, -1]]), 2, target_qubit)

    def apply_CT_gate(self, target_qubit):

        self.apply_gate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0 ,0 ,1 ,0], [0, 0, 0, cmath.exp(1j * np.pi / 4)]]), 2, target_qubit)

    def apply_CS_gate(self, target_qubit):

        self.apply_gate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0 ,0 ,1 ,0], [0, 0, 0, 1j]]), 2, target_qubit)

    def apply_CNOT_gate(self, target_qubit):
        
        self.apply_gate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), 2, target_qubit)
    
    def apply_CH_gate(self, target_qubit):
        
        unuPeRadacinaDinDoi = 1 / np.sqrt(2)
        self.apply_gate(np.array([[1, 0, 0, 0], [0, unuPeRadacinaDinDoi, 0, unuPeRadacinaDinDoi], [0, 0, 1, 0], [0, unuPeRadacinaDinDoi, 0, unuPeRadacinaDinDoi]]), 2, target_qubit)
    
    def apply_SWAP_gate(self, target_qubit):
       
        self.apply_gate(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0 ,0], [0, 0, 0, 1]]), 2, target_qubit)
    
    def apply_CNOT10_gate(self, target_qubit):
        
        self.apply_gate(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1 ,0], [0, 0, 0, 1]]), 2, target_qubit)
    
    def apply_TOFFOLI_gate(self, target_qubit):
        
        self.apply_gate(self, np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1 ,0 ,0 ,0 ,0 ,0 ,0], [0, 0, 1, 0, 0, 0, 0, 0], [0 ,0 ,0 ,1 ,0 ,0 ,0 ,0], [0 ,0 ,0 ,0 ,1 ,0 ,0 ,0], [0 ,0 ,0 ,0 ,0 ,1 ,0 ,0], [0 ,0 ,0 ,0 ,0 ,0 ,0 ,1], [0 ,0 ,0 ,0 ,0 ,0 ,1 ,0]]), 3, target_qubit)
    
    def apply_cu_gate(self, target_qubit, u_gate):
        
        cu_gate = np.zeros((4, 4), dtype=np.complex64)
        cu_gate[0:2, 0:2] = np.identity(2)
        cu_gate[2:4, 2:4] = u_gate

        self.apply_gate(cu_gate, 2, target_qubit)

    def produce_measurement(self):
        
        probabilities = self.compute_probabilities()

        # Randomly choose a state according to these probabilities
        result = np.random.choice(len(self.state), p=probabilities)

        # Convert the result to binary and pad with zeros to match the number of qubits
        result_binary = format(result, '0' + str(self.n_qubits) + 'b')

        # Convert the binary string to a list of integers
        result_list = [int(bit) for bit in result_binary]
        
        return result_list
    
    def produce_measurement_on_qubit(self, qubit):
        
        probabilities = self.compute_probabilities()

        # Randomly choose a state according to these probabilities
        result = np.random.choice(len(self.state), p=probabilities)

        # Convert the result to binary and pad with zeros to match the number of qubits
        result_binary = format(result, '0' + str(self.n_qubits) + 'b')

        # Convert the binary string to a list of integers
        result_list = [int(bit) for bit in result_binary]
        
        return result_list[qubit]
    

    def apply_phase(self, phase):

        phase_factor = cmath.exp(1j * phase)
        phased_state = [phase_factor * amplitude for amplitude in self.state]

        return phased_state

    def compute_density_matrix(self):
        density_matrix = np.outer(self.state, np.conj(self.state))

        return density_matrix

    def display_state(self):
        probabilities = self.compute_probabilities()
       
        for i in range(len(self.state)):
            print(f"State |{format(i, '0' + str(self.n_qubits) + 'b')}>: amplitude = {self.state[i]} with probability = {probabilities[i]}")

## Outside of the class functions

def quantum_fidelity_check(rho1, rho2):
    """
    Compute the quantum fidelity between two states.
    
    Parameters:
    rho1, rho2 (np.ndarray): Density matrices of the two states.

    Returns:
    float: The fidelity between rho1 and rho2.
    """

    # Compute the square root of rho1
    sqrt_rho1 = linalg.sqrtm(rho1)

    # Compute the product sqrt_rho1*rho2*sqrt_rho1
    temp = np.dot(np.dot(sqrt_rho1, rho2), sqrt_rho1)

    # Compute the square root of the above product
    sqrtm = linalg.sqrtm(temp)

    # Compute the trace of the square root
    fidelity = np.trace(sqrtm)

    return np.real(fidelity)