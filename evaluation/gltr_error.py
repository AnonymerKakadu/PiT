'''
    File name: gltr_error.py
    Author: Richard Dirauf
    Python Version: 3.8
'''


class GLTRError(Exception):
    """Custom Error for the GLTR and dataset classes.

    Attributes:
        * cl -- python class the error occured.
        * message -- explanation of the error.
    """

    def __init__(self, cl: str, message: str):
        """Create the custom error.

        Args:
            cl (str): python class the error occured
            message (str): explanation of the error
        """
        self.cl = cl
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        """Output the error as a string."""
        out = f'{self.cl} -> {self.message}'
        return out
