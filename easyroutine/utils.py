import os

def path_to_parents(levels=1):
    """
    Change the current working directory to its parent directory.
    This is equivalent to %cd ../
    
    level (int): Number of levels to go up in the directory tree.
    for example, if level=2, the function will go up two levels. (i.e. %cd ../../)
    """
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
    if levels > 1:
        for _ in range(levels-1):
            parent_dir = os.path.dirname(parent_dir)
            os.chdir(parent_dir)
    print(f"Changed working directory to: {parent_dir}")
    
    
def path_to_relative(relative_path):
    """
    Change the current working directory to a relative path.
    
    relative_path (str): The relative path to change the working directory to.
    """
    current_dir = os.getcwd()
    new_dir = os.path.join(current_dir, relative_path)
    os.chdir(new_dir)
    print(f"Changed working directory to: {new_dir}")
    
    
    
# def print_gpu_usage()
    
