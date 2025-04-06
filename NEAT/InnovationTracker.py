# --- InnovationTracker.py ---

class InnovationTracker:
    def __init__(self):
        self.innovations = {}  # (from_node_id, to_node_id) -> innovation number
        self.counter = 1

    def get_innovation(self, from_id, to_id):
        key = (from_id, to_id)
        if key not in self.innovations:
            self.innovations[key] = self.counter
            self.counter += 1
        return self.innovations[key]

# Create a global instance to import elsewhere
tracker = InnovationTracker()