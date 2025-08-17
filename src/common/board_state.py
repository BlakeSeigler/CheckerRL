import numpy as np

class StateVector:

    state: list[list[str]]

    def __init__(self):
        self.state = [
            ['o', 'b', 'o', 'b', 'o', 'b', 'o', 'b'],
            ['b', 'o', 'b', 'o', 'b', 'o', 'b', 'o'],
            ['o', 'b', 'o', 'b', 'o', 'b', 'o', 'b'],
            ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
            ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
            ['w', 'o', 'w', 'o', 'w', 'o', 'w', 'o'],
            ['o', 'w', 'o', 'w', 'o', 'w', 'o', 'w'],
            ['w', 'o', 'w', 'o', 'w', 'o', 'w', 'o'],
        ]
        self.char_dict = {
            'o': 0.0,
            'w': 1.0, 
            'b': -1.0, 
            'kw': 2.0, 
            'kb': -2.0, 
        }
        self.inv_char_dict = {v: k for k, v in self.char_dict.items()}

    @classmethod
    def from_char_numpy_vector(self, state: np.ndarray[str]) -> bool:
        assert state.shape == (64,)
        try:
            self.state = state.reshape(8,8)
            return True
        except:
            return False

    def to_char_numpy_array(self) -> np.ndarray[str]:
        return np.array(self.state).flatten()
    
    @classmethod
    def from_float_numpy_vector(self, state: np.ndarray[float]) -> bool:
        assert state.shape == (64,)
        try:
            rounded = np.round(np.array(state))
            intermediate = np.array([self.inv_char_dict[f] for f in rounded])     
            self.state = intermediate.reshape(8,8)
            return True
        except:
            return False

    def to_int_numpy_array(self) -> np.ndarray[float]:
        intermediate = np.array(self.state).flatten()
        return np.array([self.char_dict[f] for f in intermediate])
    
    def __str__(self):
        text = ""
        for line in self.state:
            text += f"{line}" + "\n "

        return text