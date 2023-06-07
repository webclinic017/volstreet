class ApiKeyNotFound(Exception):
    """Exception raised for missing API Key."""
    def __init__(self, message="API Key not found"):
        self.message = message
        super().__init__(self.message)
