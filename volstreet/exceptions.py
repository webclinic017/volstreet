class VolStreetException(Exception):
    """Base class for other exceptions."""

    def __init__(self, message="VolStreet Exception", code=500):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ApiKeyNotFound(VolStreetException):
    """Exception raised for missing API Key in the environment variables."""

    def __init__(self, message="API Key not found"):
        super().__init__(message, code=404)


class OptionModelInputError(VolStreetException):
    """Exception raised for errors in the inputs for Black Scholes model."""

    def __init__(self, message="Invalid inputs for Black Scholes model"):
        super().__init__(message, code=3500)
