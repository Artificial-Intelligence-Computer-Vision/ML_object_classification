from header_imports import *
from computer_vision_model_training import *

if __name__ == "__main__":
    
    # Begin analysis for building model or training it
    if len(sys.argv) != 1:
        if sys.argv[1] == "model_building":
            computer_vision__analysis_obj = computer_vision_building(model_type = sys.argv[2], image_type = sys.argv[3])

        # Seperate images base on names
        if sys.argv[1] == "model_training":
            computer_vision_analysis_obj = computer_vision_training(model_type = sys.argv[2], image_type = sys.argv[3])
