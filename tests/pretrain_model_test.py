import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.TrainedMusicGen import TrainedMusicGen

def test_trained_model_initialization():
    model = TrainedMusicGen()
    assert model is not None
    print("TrainedMusicGen initialized successfully.")

if __name__ == "__main__":
    test_trained_model_initialization()