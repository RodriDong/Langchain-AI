from langgraph.checkpoint.memory import MemorySaver
import json
import os

class WritingMemory:
    def __init__(self):
        # Khoi tao bo nho dung MemorySaver cua langgraph
        self.memory = MemorySaver()
        self.history_file = "writing_history.json"
        self.history = self._load_history()

    def _load_history(self):
        # Doc lich su tu file neu co
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_evaluation(self, writing: str, feedback: str, score: float):
        # Luu ket qua cham vao file
        entry = {
            "writing_snippet": writing[:200],
            "feedback_summary": feedback,
            "score": score
        }
        self.history.append(entry)
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def get_history_summary(self) -> str:
        # Tra ve tom tat lich su cham bai
        if not self.history:
            return "Chua co bai nao duoc cham truoc do."

        scores = [h["score"] for h in self.history if h.get("score")]
        avg = round(sum(scores) / len(scores), 1) if scores else 0

        return (
            f"Da cham {len(self.history)} bai. "
            f"Diem trung binh: {avg}/10. "
            f"Diem bai gan nhat: {self.history[-1].get('score', 'N/A')}/10."
        )

    def get_memory(self):
        # Tra ve doi tuong MemorySaver de dung voi langgraph agent
        return self.memory