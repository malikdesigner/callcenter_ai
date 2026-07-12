"""
Dialogue Policy Layer

Acts as the central state machine and memory layer.
Decides what action should be taken next based on extracted intent and current state.
"""

from loguru import logger

class DialoguePolicy:
    def __init__(self, language: str = "ur"):
        self.language = language
        self.state = "GREETING"
        self.retry_count = 0
        self.last_intent = None
        self.collected_data = {}
        
        # Hardcoded fallbacks for extreme garbage to bypass LLM
        self.hard_fallbacks = {
            "ur": "سوری، آواز واضح نہیں ہے۔ کیا آپ دہرا سکتے ہیں؟",
            "ro": "Sorry, awaaz wazeh nahi hai. Kya aap dohra sakte hain?",
            "en": "Sorry, I couldn't hear that clearly. Could you repeat?"
        }

    def evaluate(self, user_intent: str, extracted_data: dict) -> dict:
        """
        Evaluate the user intent and update state.
        
        Args:
            user_intent: Classified intent of the user (e.g., CONFUSED, QUESTIONING, SYMPTOM, ANSWERING, GARBAGE)
            extracted_data: Information gathered from the user message.
            
        Returns:
            dict containing instructions for the NLG (LLM) layer:
            {
                "current_state": str,
                "nlg_constraint": str,  # Instructions for how LLM should respond
                "is_hard_fallback": bool,
                "fallback_text": str
            }
        """
        logger.debug(f"[Policy] Evaluating intent: {user_intent} in state: {self.state}")
        
        # Merge newly extracted data
        for k, v in extracted_data.items():
            if v:
                self.collected_data[k] = v

        # 1. Hard Fallbacks for Garbage
        if user_intent == "GARBAGE":
            self.retry_count += 1
            if self.retry_count > 3:
                # Give up or escalate
                return {
                    "current_state": "END",
                    "nlg_constraint": "Apologize and end the call due to poor audio.",
                    "is_hard_fallback": False,
                    "fallback_text": ""
                }
            return {
                "current_state": self.state,
                "nlg_constraint": "hard_fallback",
                "is_hard_fallback": True,
                "fallback_text": self.hard_fallbacks.get(self.language, self.hard_fallbacks["en"])
            }

        # 2. Soft Fallbacks for ambiguous intents
        if user_intent in ["CONFUSED", "QUESTIONING"]:
            self.retry_count += 1
            return {
                "current_state": self.state, # Do not advance state
                "nlg_constraint": f"The user is {user_intent.lower()}. Do not rigidly ask for the slot. Enter natural conversation mode to answer or clarify, then gently bring them back to asking for the required info for {self.state}.",
                "is_hard_fallback": False,
                "fallback_text": ""
            }

        # 3. State Advancement Logic
        self.retry_count = 0  # Reset on valid progress
        
        if user_intent == "CORRECTING":
            # Just stay in current state and let LLM know they corrected something
            return {
                "current_state": self.state,
                "nlg_constraint": "Acknowledge the correction gracefully and update the information.",
                "is_hard_fallback": False,
                "fallback_text": ""
            }

        # Determine next state based on missing fields
        next_state = self._determine_next_state()
        
        nlg_constraint = ""
        if next_state == self.state and user_intent == "ANSWERING":
            # They answered but we didn't get the required info (LLM extraction failed or user dodged)
            self.retry_count += 1
            nlg_constraint = f"User hasn't provided the required info for {self.state} yet. Try asking differently in a natural way. (Retry #{self.retry_count})"
        
        self.state = next_state
        return {
            "current_state": self.state,
            "nlg_constraint": nlg_constraint,
            "is_hard_fallback": False,
            "fallback_text": ""
        }
        
    def _determine_next_state(self) -> str:
        """State machine progression based on collected data."""
        if not self.collected_data.get("patient_name"):
            return "ASK_NAME"
        if not self.collected_data.get("patient_phone"):
            return "ASK_PHONE"
        if not self.collected_data.get("department"):
            return "ASK_SYMPTOM"
        if not self.collected_data.get("doctor_name"):
            return "ASK_DOCTOR"
        if not self.collected_data.get("appointment_date"):
            return "ASK_DATE"
        if not self.collected_data.get("appointment_time"):
            return "ASK_TIME"
        
        return "CONFIRMATION"

    def sync_from_collected(self, collected: dict) -> None:
        """
        Re-sync policy state from the agent's latest collected data.
        Call at the start of every turn so the fast_engine's advances
        (phone extraction, name extraction, etc.) are visible to the
        semantic validator and intent detector before they run.
        """
        for k, v in collected.items():
            if v:
                self.collected_data[k] = v
        new_state = self._determine_next_state()
        if new_state != self.state:
            # State advanced — clear accumulated retry counter so the next
            # valid answer isn't penalised for previous retries in the old state
            self.retry_count = 0
        self.state = new_state

    def reset(self):
        self.state = "GREETING"
        self.retry_count = 0
        self.collected_data = {}
