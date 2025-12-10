import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.MLPMusicGen import MLPMusicGen

class TrainedMusicGen(MLPMusicGen):
    """
    Creating a model with 100 hidden units and loading pre-trained parameters
    """
    def __init__(self, model_path = "Models/param.txt"):
        super().__init__(context_length=20, hidden_sizes= [100], activation_type="relu")
        self.load_model(model_path)
    
    def load_model(self, model_path ):
        curr_param = self.parameters()

        with open(model_path, "r") as f:
            pre_trained_params = [float(line.strip()) for line in f]
        try: 
            for i, p in enumerate(curr_param):
                p.data = pre_trained_params[i]
        except Exception as e:
            print(len(pre_trained_params), len(list(curr_param)))
        else: 
            print("Model parameters loaded successfully.")

if __name__ == "__main__":
    model = TrainedMusicGen("Models/param.txt")
    
    