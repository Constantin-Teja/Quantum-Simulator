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

    def test_create_gate_dimension(self):
        with self.assertRaises(AssertionError): Gate(1)
        with self.assertRaises(AssertionError): Gate(np.array([1]))

    def test_create_gate_shape(self):
        with self.assertRaises(AssertionError): Gate([1, 2, 3])
        with self.assertRaises(AssertionError): Gate(np.array([[1, 0], [1, 0], [1, 0]]))
        
    def test_create_gate_unitary(self):
        with self.assertRaises(AssertionError): Gate(np.array([[1, 1], [1, 1]]))

    def test_create_gate_tensor_result(self):
        with self.assertRaises(AssertionError): Gate(np.identity(3))


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


class Quantum_Helper_Tests(unittest.TestCase):
    
    def test_produce_measurement(self):
        state_1 = State(1)
        state_2 = State(2)
        state_3 = State(1)

        state_2.initialize_state(np.array([1, 1]))
        
        state_3.initialize_state([1])
        state_3.apply_H_gate(0)

        self.assertTrue(np.allclose(qh.produce_measurement(state_1.state, 1, "COMPUTATIONAL"), np.array([[0]])))
        self.assertTrue(np.allclose(qh.produce_measurement(state_2.state, 2, "COMPUTATIONAL"), np.array([[1], [1]])))
        self.assertTrue(np.allclose(qh.produce_measurement(state_3.state, 1, "HADAMARD"), np.array([[0]])))

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
        

# Main entry point to run the tests
if __name__ == '__main__':
    unittest.main()
