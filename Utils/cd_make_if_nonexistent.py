import os


def cd_make_if_nonexistent(directory):
    """
    Change to a directory if it exists. If it does not, make it then change to it.
    :param directory: string. Directory to change to.
    """
    if not os.path.exists(directory):
        print(f'Making directory: {directory}')
        os.mkdir(directory)
    print(f'Changing to directory: {directory}')
    os.chdir(directory)
