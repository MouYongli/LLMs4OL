import os
import glob

def find_root_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(current_dir, 'requirements.txt')):  # find root path by searching for requirements.txt file
        current_dir = os.path.dirname(current_dir)
    return current_dir


def find_trained_model_path(path_pattern):
    result = ""
    try:
        file_paths = glob.glob(path_pattern)
        result = file_paths[0]
    except (IndexError):
        print("Check your Model Path!")

    return result 