class ConversationMemory:
    def __init__(self, max_memory_size=10):
        self.memory = []
        self.max_memory_size = max_memory_size

    def add(self, user_input, summary, facts):
        """Add a user-AI interaction to memory with query type metadata."""
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0)  # Remove the oldest interaction
        self.memory.append({"User": user_input, "summary": summary, "facts": facts})

    def get_history(self):
        """Retrieve the conversation history filtered by query type."""
        return self.memory