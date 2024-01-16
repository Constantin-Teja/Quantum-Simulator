import numpy as np
import scipy.linalg as linalg

# Static class
class Quantum_Helper:
    # This class contains helper methods for quantum systems

    __qubit_basis = {
        # Each base vectors are represented on lines
        "COMPUTATIONAL" : np.array([ [1, 0], [0, 1] ]),
        "HADAMARD" : (1/np.sqrt(2)) * np.array([ [1, -1], [1, 1] ])
    }

    # Sqrt(2) is used multiple times and is heavy
    sqrt2 = np.sqrt(2)
    
    @staticmethod
    def get_extended_basis(base_name, n_qubits: np.uint64):
        basis = Quantum_Helper.__qubit_basis[base_name]
        extended_basis = basis
        for i in range(n_qubits - 1):
            extended_basis = np.kron(extended_basis, basis)
            
        return extended_basis
    
    @staticmethod
    def compute_adjoint(matrix):
        return matrix.conjugate().transpose()
    
    @staticmethod
    def get_base(base_name: str):
        return Quantum_Helper.__qubit_basis[base_name]
    
    @staticmethod
    def compute_measurement_matrix(pure_state):
        return np.dot(pure_state, Quantum_Helper.compute_adjoint(pure_state))

    @staticmethod
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