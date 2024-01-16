from Circuit import Circuit, import_gate, export_gate
from State import State
from Transformation_Enum import Transformation as Tr
from Gate import Gate

if __name__ == '__main__':

    # circuit = Circuit(3)
    # circuit.apply_transformation(Tr.H, [1, 0])
    
    # circuit.apply_transformation(Tr.X, [1, 1])

    #circuit.export_circuit("circuit.json")

    circuit = Circuit.import_circuit("circuit.json")

    #circuit.system_state.display_state()

    circuit.run()

    export_gate(Gate.gates["X"], "x_gate.py")

    x = import_gate("x_gate.py")

    print(x.get_gate())



    #circuit.system_state.display_state()