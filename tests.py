import unittest
import numpy as np

from Gate import Gate
from State import State
from Quantum_Helper import Quantum_Helper as qh

class Gate_Tests(unittest.TestCase):
        
    def test_create_gate_success(self):
        self.assertTrue(np.array_equal(Gate(np.identity(2)).get_gate(), np.identity(2)))
        self.assertTrue(np.array_equal(Gate(np.identity(4)).get_gate(), np.identity(4)))
        self.assertTrue(np.array_equal(Gate(np.identity(8)).get_gate(), np.identity(8)))
        self.assertTrue(np.allclose(Gate(np.array([[1, 1], [1, -1]]) / np.sqrt(2)).get_gate(), Gate.gates["H"].get_gate()))

    def test_create_by_composing_success(self):
        self.assertTrue(np.allclose(Gate.create_by_composing([Gate.gates["X"].get_gate(), Gate.gates["X"].get_gate()]).get_gate(), np.identity(2)))

    def test_create_gate_dimension_error(self):
        with self.assertRaises(AssertionError): Gate(1)
        with self.assertRaises(AssertionError): Gate(np.array([1]))

    def test_create_gate_shape_error(self):
        with self.assertRaises(AssertionError): Gate([1, 2, 3])
        with self.assertRaises(AssertionError): Gate(np.array([[1, 0], [1, 0], [1, 0]]))
        
    def test_create_gate_unitary_error(self):
        with self.assertRaises(AssertionError): Gate(np.array([[1, 1], [1, 1]]))

    def test_create_gate_tensor_result_error(self):
        with self.assertRaises(AssertionError): Gate(np.identity(3))

    def test_create_by_composing_error(self):
        with self.assertRaises(AssertionError): Gate.create_by_composing([np.identity(2), np.array([ [3, 1], [-1, 2] ])])


class State_Tests(unittest.TestCase):

    def test_create_state(self):
        self.assertTrue(np.array_equal(State(2).state, np.array([[1], [0], [0], [0]])))
        self.assertTrue(np.array_equal(State(3).state, np.array([[1], [0], [0], [0], [0], [0], [0], [0]])))

    def test_initialize_state_success(self):
        state_1 = State(2)
        state_2 = State(3)

        state_1.initialize_state(np.array([1, 1]))
        state_2.initialize_state(np.array([1, 1, 1]))

        self.assertTrue(np.array_equal(state_1.state, np.array([[0], [0], [0], [1]])))
        self.assertTrue(np.array_equal(state_2.state, np.array([[0], [0], [0], [0], [0], [0], [0], [1]])))

    def test_intialize_state_fail(self):
        with self.assertRaises(AssertionError): State(3).initialize_state(np.array([1, 1]))

    def test_apply_H_gate(self):
        state = State(2)
        state.apply_H_gate(0)
        state.apply_H_gate(1)

        self.assertTrue(np.allclose(state.state, np.array([[0.5], [0.5], [0.5], [0.5]])))

    def test_apply_X_gate(self):
        state = State(1)
        state.apply_X_gate(0)

        self.assertTrue(np.allclose(state.state, np.array([[0], [1]])))

class Measurement_Tests(unittest.TestCase):
    
    def test_produce_measurement(self):
        state_1 = State(1)
        state_2 = State(2)
        state_3 = State(1)

        state_2.initialize_state(np.array([1, 1]))
        
        state_3.initialize_state([1])
        state_3.apply_H_gate(0)

        self.assertTrue(np.allclose(state_1.produce_measurement(), np.array([[0]])))
        self.assertTrue(np.allclose(state_2.produce_measurement("COMPUTATIONAL"), np.array([[1], [1]])))
        self.assertTrue(np.allclose(state_3.produce_measurement("HADAMARD"), np.array([[0]])))

class Density_Operator_Tests(unittest.TestCase):

    def test_produce_density_matrix(self):
        state = State(2)

        self.assertTrue(np.allclose(state.compute_density_matrix(), np.array([ [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0] ])))
        self.assertTrue(np.allclose(state.compute_density_matrix("HADAMARD"), np.array([ [0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, 0.25] ])))

        state.initialize_state([1, 1])

        self.assertTrue(np.allclose(state.compute_density_matrix(), np.array([ [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1] ])))
        self.assertTrue(np.allclose(state.compute_density_matrix("HADAMARD"), np.array([ [0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, 0.25] ])))

        state.apply_H_gate(0)
        state.apply_H_gate(1)
        
        self.assertTrue(np.allclose(state.compute_density_matrix(), np.array([ [0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, 0.25] ])))
        self.assertTrue(np.allclose(state.compute_density_matrix("HADAMARD"), np.array([ [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0] ])))

    def test_quantum_fidelity(self):
        state = State(2)
        rho1 = state.compute_density_matrix()

        state.initialize_state(np.array([1, 1]))
        rho2 = state.compute_density_matrix()

        self.assertEqual(qh.quantum_fidelity_check(rho1, rho2), 0)
        self.assertEqual(qh.quantum_fidelity_check(rho1, rho1), 1)
        

# Main entry point to run the tests
if __name__ == '__main__':
    unittest.main()
