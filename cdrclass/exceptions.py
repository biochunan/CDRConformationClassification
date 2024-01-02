"""
Custom defined errors
"""
class ClassifierNotExistError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
