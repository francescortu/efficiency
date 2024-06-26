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