from abc import ABC


class AbstractLLMClient(ABC):
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
