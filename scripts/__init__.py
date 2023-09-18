"""
This file allows us to import scripts as if it were a package. It currently only has the main
function to make things run, but you can import other things as if the other files were subpackages
"""

try:
    from .main_code import main as main
except ImportError:
    from main_code import main as main