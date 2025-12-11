import os
import sys



from composerml.models import TrainedMusicGen

def test_trained_model_initialization():
    model = TrainedMusicGen()
    assert model is not None
    print("TrainedMusicGen initialized successfully.")

if __name__ == "__main__":
    test_trained_model_initialization()