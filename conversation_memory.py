import json, os
from datetime import datetime

class ConversationMemory:
    def __init__(self, max_memory_size=30, file_path="history.json"):
        self.file_path = file_path  # Fix: Store file path
        self.max_memory_size = max_memory_size

        # Ensure history.json exists
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f)

    def _load_memory(self):
        """Load conversation history from the JSON file."""
        with open(self.file_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []  # Return empty list if file is empty or corrupted
    
    def _save_memory(self, memory):
        """Save updated conversation history to the JSON file."""
        with open(self.file_path, "w") as f:
            json.dump(memory, f, indent=4)
    
    def add(self, user_input, summary, facts):
        """Add a user-AI interaction to memory with query type metadata."""
        memory = self._load_memory()  # Fix: Load memory from file before modifying

        if len(memory) >= self.max_memory_size:
            memory.pop(0)  # Remove the oldest interaction

        memory.append({"User": user_input, "summary": summary, "facts": facts, "timestamp": datetime.now().isoformat()})
        self._save_memory(memory)

    def get_history(self):
        """Retrieve the conversation history filtered by query type."""
        return self._load_memory()  # Fix: Always fetch latest memory