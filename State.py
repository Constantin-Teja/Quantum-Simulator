import numpy as np
from Quantum_Helper import Quantum_Helper as qh
from Gate import Gate
import cmath

class State:
    # This class should contain methods which have effect upon state
    # or is an intrisec property

    def __init__(self, n_qubits: np.uint16):
        self.n_qubits = n_qubits # TODO: make private
        self.state = np.zeros((2**n_qubits, 1), dtype=np.complex64)
        self.state[0] = 1 # Set |00...0> as default state

    def initialize_state(self, qubit_values):
        assert len(qubit_values) == self.n_qubits
        self.state = np.zeros((2**len(qubit_values), 1), dtype=np.complex64)
        sum_of_coefficients = 0
        for i in range(len(qubit_values)):
            sum_of_coefficients = sum_of_coefficients + qubit_values[i] * 2**(len(qubit_values) - i - 1)
        self.state[sum_of_coefficients][0] = 1

    def apply_gate(self, gate, n_gate, starting_qubit):
        applied_gate = np.identity(1)

        for i in range(starting_qubit):
            applied_gate = np.kron(applied_gate, np.identity(2))
        
        applied_gate = np.kron(applied_gate, gate.get_gate())

        for i in range(starting_qubit + n_gate, self.n_qubits):
            applied_gate = np.kron(applied_gate, np.identity(2))

        self.state = np.dot(applied_gate, self.state)
       
    def apply_H_gate(self, target_qubit):
        self.apply_gate(Gate.gates["H"], 1, target_qubit)
    
    def apply_X_gate(self, target_qubit):
        self.apply_gate(Gate.gates["X"], 1, target_qubit)
    
    # TODO: unittest
    def apply_S_gate(self, target_qubit):
        self.apply_gate(Gate.gates["S"], 1, target_qubit)

    # TODO: unittest
    def apply_CNOT_gate(self, target_qubit):
        self.apply_gate(Gate.gates["CNOT"], 2, target_qubit)
    
    # TODO: unittest
    def apply_CH_gate(self, target_qubit):
        self.apply_gate(Gate.gates["CH"], 2, target_qubit)
    
    # TODO: unittest
    def apply_SWAP_gate(self, target_qubit):
        self.apply_gate(Gate.gates["SWAP"], 2, target_qubit)
    
    # TODO: unittest
    def apply_CNOT10_gate(self, target_qubit):
        self.apply_gate(Gate.gates["CNOT10"], 2, target_qubit)
    
    # TODO: unittest
    def apply_TOFFOLI_gate(self, target_qubit):
        self.apply_gate(Gate.gates["TOFFOLI"], 3, target_qubit)

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

    def apply_CU_gate(self, target_qubit, U_gate):
        
        CU_gate = np.zeros((4, 4), dtype=np.complex64)
        CU_gate[0:2, 0:2] = np.identity(2)
        CU_gate[2:4, 2:4] = U_gate

        self.apply_gate(CU_gate, 2, target_qubit)

    def apply_P_gate(self, target_qubit, phi):
        self.apply_gate(np.array([[1, 0], [0, np.exp(1j * phi)]]), 1, target_qubit)
    
    def apply_Z_gate(self, target_qubit):
        self.apply_gate(np.array([[1, 0], [0, -1]]), 1, target_qubit)

    def apply_S_gate(self, target_qubit):
        self.apply_gate(np.array([[1, 0], [0, np.exp((1j * np.pi)/2)]]), 1, target_qubit)

    def apply_S_dag_gate(self, target_qubit):
        self.apply_gate(np.array([[1, 0], [0, np.exp(-(1j * np.pi)/2)]]), 1, target_qubit)

    def apply_T_gate(self, target_qubit):
        self.apply_gate(np.array([[1, 0], [0, np.exp((1j * np.pi)/4)]]), 1, target_qubit)
    
    def apply_T_dag_gate(self, target_qubit):
        self.apply_gate(np.array([[1, 0], [0, np.exp(-(1j * np.pi)/4)]]), 1, target_qubit)

    def apply_RX_gate(self, target_qubit, theta):
        self.apply_gate(np.array([[np.cos(theta/2), -1j * np.sin(theta/2)], [-1j * np.sin(theta/2), np.cos(theta/2) ]]), 1, target_qubit)

    def apply_RY_gate(self, target_qubit, theta):
        self.apply_gate(np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]]), 1, target_qubit)

    def apply_RZ_gate(self, target_qubit, theta):
        self.apply_gate(np.array([[np.exp(-1j * (theta/2)), 0], [0, np.exp(1j * (theta/2))]]), 1, target_qubit)

    def apply_SX_gate(self, target_qubit):
        self.apply_gate(np.array([[(1 + 1j)/2, (1 - 1j)/2], [(1 - 1j)/2, (1 + 1j)/2]]), 1, target_qubit)

    def apply_SX_dag_gate(self, target_qubit):
        self.apply_gate(np.array([[(1 - 1j)/2, (1 + 1j)/2], [(1 + 1j)/2, (1 - 1j)/2]]), 1, target_qubit)


    # TODO: unittest
    def calculate_state_amplitudes(self):
        # Check if the input state vector is normalized (sum of squared amplitudes is close to 1)
        norm = np.linalg.norm(self.state)
        if not np.isclose(norm, 1.0):
            raise ValueError("Input state vector must be normalized.")

        n_qubits = int(np.log2(len(self.state)))

        # Initialize a dictionary to store the amplitudes of all basis states
        state_amplitudes = {}

        # Iterate through all possible basis states and calculate their amplitudes
        for i in range(2 ** n_qubits):
            basis_state = np.binary_repr(i, width=n_qubits)  # Convert the integer to binary representation
            amplitude = self.state[i]
            state_amplitudes[basis_state] = amplitude

        return state_amplitudes
    
    def produce_measurement_on_qubit(self, qubit, base_name="COMPUTATIONAL"):
        return self.produce_measurement(base_name)[qubit]
    
    # TODO: improve. Add unittests
    def get_phased_state(self, phase):
        phase_factor = cmath.exp(1j * phase)
        phased_state = [phase_factor * amplitude for amplitude in self.state]
        return phased_state

    def compute_density_matrix(self, base_name = "COMPUTATIONAL"):
        probabilities = self.compute_probabilities(base_name)
        extended_base = qh.get_extended_basis("COMPUTATIONAL", self.n_qubits)

        density_matrix = np.zeros((len(self.state), len(self.state)))
        for i in range(len(extended_base)):
            density_matrix = density_matrix + probabilities[i] * np.outer(qh.compute_adjoint(extended_base[i]), extended_base[i])

        return density_matrix
    
    def display_state(self):
        probabilities = self.compute_probabilities()      
        for i in range(len(self.state)):
            print(f"State |{format(i, '0' + str(self.n_qubits) + 'b')}>: amplitude = {self.state[i]} with probability = {probabilities[i]}")

    def display_state_amplitudes(self):
        state_amplitudes = self.calculate_state_amplitudes()
        # Display the state amplitudes
        for basis_state, amplitude in state_amplitudes.items():
            print(f"|{basis_state}>: {amplitude}")

    def compute_probabilities(self, base_name = "COMPUTATIONAL"):
        # Initialize probabilities array
        probabilities = np.zeros(len(self.state))

        # Compute probabilities of states in base_name
        basis_measurement = qh.get_extended_basis(base_name, self.n_qubits)

        for i in range(len(basis_measurement)):
            base = np.array([basis_measurement[i]]).transpose()
            Mi = qh.compute_measurement_matrix(base)
            temp = np.dot(Mi, self.state)
            probabilities[i] = np.dot(qh.compute_adjoint(self.state), temp).real

        return probabilities
    
    def produce_measurement(self, base_name = "COMPUTATIONAL"):
        probabilities = self.compute_probabilities(base_name)
        
        # Randomly choose a state in the specified base according to these probabilities
        measured_state_index = np.random.choice(len(self.state), p=probabilities)
        
        # Convert the result to binary and pad with zeros to match the number of qubits
        result_binary = format(measured_state_index, '0' + str(self.n_qubits) + 'b')

        # Convert the binary string to a list of integers
        result_list = [int(bit) for bit in result_binary]
        
        return result_list
    
    def calculate_state_amplitudes(self):
        # Check if the input state vector is normalized (sum of squared amplitudes is close to 1)
        norm = np.linalg.norm(self.state)
        if not np.isclose(norm, 1.0):
            raise ValueError("Input state vector must be normalized.")

        # Initialize a dictionary to store the amplitudes of all basis states
        state_amplitudes = {}

        # Iterate through all possible basis states and calculate their amplitudes
        for i in range(2 ** self.n_qubits):
            basis_state = np.binary_repr(i, width=self.n_qubits)  # Convert the integer to binary representation
            amplitude = self.state[i]
            state_amplitudes[basis_state] = amplitude

        return state_amplitudes
    
    # Same as above but returns an array of values without the keys
    def get_amplitudes(self):
        return self.state
    
    # QST: Collapse purpose. For future applications. TODO: Move in State when is done
    def collapse_state(self, probabilities, index, base_name):
        # Compute probabilities of states in base_name
        basis_measurement = qh.get_extended_basis(base_name, self.n_qubits)
        base = np.array([basis_measurement[index]]).transpose()
        Mm = qh.compute_measurement_matrix(base)
        collapsed_state = np.dot(Mm, self.state) / (np.sqrt(probabilities[index]))
        return collapsed_state