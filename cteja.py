from Circuit import Circuit, import_gate, export_gate
from State import State
from Transformation_Enum import Transformation as Tr
from Gate import Gate

if __name__ == '__main__':
    state = State(2)
    state.apply_H_gate(0)
    print(state.compute_probabilities())
    state.apply_H_gate(0)
    print(state.compute_probabilities())