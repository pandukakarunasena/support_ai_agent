
class MCPToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_tool_response(self) -> dict:
        """
        Convert to a dictionary that the MCP‐client library will send
        back to your front‐end. Your actual server might require a specific
        class or JSON shape; adapt as needed.
        """
        return {
            "success":      False,
            "error_code":   self.code,
            "error_message": self.message,
            "result":       None
        }
