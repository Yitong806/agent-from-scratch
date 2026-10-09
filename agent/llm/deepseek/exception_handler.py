from agent.llm.exception import LLMException

INVALID_FORMAT = 400
AUTH_FAILED = 401
INSUFFICIENT_BALANCE = 402
INVALID_PARAMETER = 422
RATE_LIMIT_REACHED = 429
SERVER_ERROR = 500
SERVER_OVERLOADED = 503

UNKNOWN_ERROR = "Unknown Error"

exception_message_mapping = {
    INVALID_FORMAT: """
    Cause: Invalid request body format.
    Solution: Please modify your request body according to the hints in the error message. For more API format details, please refer to DeepSeek API Docs.
    """,

    AUTH_FAILED: """
    Cause: Authentication fails due to the wrong API key.
    Solution: Please check your API key. If you don't have one, please create an API key first.
    """,

    INSUFFICIENT_BALANCE: """
    Cause: You have run out of balance.
    Solution: Please check your account's balance, and go to the Top up page to add funds.
    """,

    INVALID_PARAMETER: """
    Cause: Your request contains invalid parameters.
    Solution: Please modify your request parameters according to the hints in the error message. For more API format details, please refer to DeepSeek API Docs.
    """,

    RATE_LIMIT_REACHED: """
    Cause: You are sending requests too quickly.
    Solution: Please pace your requests reasonably. We also advise users to temporarily switch to the APIs of alternative LLM service providers, like OpenAI.
    """,

    SERVER_ERROR: """
    Cause: Our server encounters an issue.
    Solution: Please retry your request after a brief wait and contact us if the issue persists.
    """,

    SERVER_OVERLOADED: """
    Cause: The server is overloaded due to high traffic.
    Solution: Please retry your request after a brief wait.
    """
}

class DeepSeekException(LLMException):

    def __init__(self, code: int, message: str):
        self.code = code
        super().__init__(message)

def deepseek_exception(exception_code: int, exception_message: str) -> Exception:
    return DeepSeekException(code=exception_code,
                            message=exception_message_mapping.get(exception_code,
                                                                  UNKNOWN_ERROR)+' '+ exception_message)
