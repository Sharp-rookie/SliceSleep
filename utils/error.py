# -*- coding: utf-8 -*-
__all__ = [
    'SliceNumError',
    'UENumError',
    'UETypeError',
    ]

class Error(Exception):
    """Base class for exceptions in GNB
    """

    pass

class SliceNumError(Error):
    """Exception raised for unsupported slice num

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class UENumError(Error):
    """Exception raised for unsupported UE num

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class UETypeError(Error):
    """Exception raised for unsupported UE type

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message