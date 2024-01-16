import numpy as np
import json

from Transformation_Enum import Transformation as Tr
from State import State
from Gate import Gate

class Circuit:
    def __init__(self, n_qubits: np.uint16, transformations=None, input_state=None, system_state=None):
        self.n_qubits = n_qubits
        self.transformations = transformations if transformations is not None else []
        self.input_state = input_state if input_state is not None else State(n_qubits)
        self.system_state = system_state if system_state is not None else State(n_qubits)

    def reset(self):
        self.system_state = self.input_state
        self.input_state = State(self.n_qubits)
        self.input_state.state = self.system_state.state

    def set_input(self, input_state):
        self.system_state = input_state

    def run(self):
        for transf, args in self.transformations:
            self.system_state.apply_gate(Gate.gates[transf], args[0], args[1])

    def apply_transformation(self, transformation: Tr, args):
        self.transformations.append([transformation, args])

    def export_circuit(self, json_name):
        serialized_data = serialize_data(self)
        
        with open(json_name, "w") as file:
            file.write(serialized_data)

    @staticmethod
    def import_circuit(json_name):
        json_object = None
        with open(json_name, "r") as file:
            json_object = file.read()

        return deserialize_data(json_object)
    

# TODO Move in a new file
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (State, Gate, Circuit)):
            return obj.__dict__
        if isinstance(obj, Tr):
            return obj.name
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        
        return super().default(obj)
 
def object_hook(dct):
    if isinstance(dct, (State)):
        return dct

    if '__complex__' in dct:
        return complex(dct['real'], dct['imag'])
    elif 'n_qubits' in dct and 'transformations' in dct and 'input_state' in dct and 'system_state':
        n_qubits = int(dct['n_qubits'])
        transformations = [object_hook(st) for st in dct["transformations"]]
        input_state = object_hook(dct['input_state'])
        system_state = object_hook(dct['system_state'])
        return Circuit(n_qubits=n_qubits, transformations=transformations, input_state=input_state, system_state=system_state)
    elif 'n_qubits' in dct and 'state' in dct:
        n_qubits = int(dct['n_qubits'])
        state = [object_hook(st) for st in dct["state"]]
        return State(n_qubits=n_qubits, state=np.array(state))

    return dct
    
def serialize_data(obj):
    return json.dumps(obj, cls=CustomJSONEncoder, indent=2)

def deserialize_data(json_data):
    return json.loads(json_data, object_hook=object_hook)

# # Serialize to JSON
# serialized_data = json.dumps(obj.__dict__)

# # Writing to a file (optional)
# with open("serialized_data.json", "w") as file:
#     file.write(serialized_data)

# # Deserialize from JSON
# json_data = json.loads(serialized_data)
# reconstructed_obj = MyClass(**json_data)
        
        # param1: pe cati qubiti e transf
        # param2: de la care qubit se inceapa