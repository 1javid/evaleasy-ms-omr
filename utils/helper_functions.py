import shutil
import tempfile

def create_temp_output_dir():
    """
    Create a temporary directory for storing output files.
    """
    return tempfile.mkdtemp()

def cleanup_temp_output_dir(output_dir):
    """
    Clean up the temporary output directory.
    """
    shutil.rmtree(output_dir)

def sort_bubbles(bubbles, sort_by_row=True):
    """
    Sort bubbles by their coordinates, either by row or column.
    """
    if sort_by_row:
        return sorted(bubbles, key=lambda bubble: (bubble[1], bubble[0]))
    return sorted(bubbles, key=lambda bubble: (bubble[0], bubble[1]))
