
class PCIMockLogic:
    """
    Simulates Predictive Customer Intelligence (PCI) logic.
    Provides mock customer segmentation and product/offer suggestions based on simple rules.
    """

    def __init__(self):
        pass

    def get_customer_segment(self, user_query: str, chat_history_str: str = "") -> str:
        """
        Determines a mock customer segment based on the query and chat history.
        This is a simplified, rule-based segmentation.
        """
        query_lower = user_query.lower()
        history_lower = chat_history_str.lower()

        if "cancel" in query_lower or "unsubscri" in query_lower or "leave" in history_lower:
            return "churn_risk"
        elif "price" in query_lower or "discount" in query_lower or "deal" in query_lower:
            return "price_sensitive"
        elif "premium" in query_lower or "upgrade" in query_lower or "high-end" in history_lower:
            return "high_value_prospect"
        elif "new user" in history_lower or "first time" in history_lower or "how to start" in query_lower:
            return "new_customer"
        else:
            return "standard"

    def get_suggestion(self, customer_segment: str) -> str:
        """
        Provides a product, offer, or response suggestion based on the customer segment.
        """
        if customer_segment == "churn_risk":
            return "We understand your concern. Would you like to speak with a retention specialist or would a 20% discount on your next month's service help?"
        elif customer_segment == "price_sensitive":
            return "We have several budget-friendly options available. Would you be interested in our 'Basic Plan' or current promotional deals?"
        elif customer_segment == "high_value_prospect":
            return "Based on your interest, our 'Premium Plus' package offers exclusive features and priority support. Would you like to know more?"
        elif customer_segment == "new_customer":
            return "Welcome! To help you get started, we recommend exploring our interactive tutorial or checking out our 'Quick Start Guide'."
        elif customer_segment == "standard":
            return "I can help with that. What specific information or product are you looking for?"
        else:
            return "I'm not sure what to suggest based on that segment, but I'm here to help with your query."

# Example usage for testing (optional, for direct testing of this module)
if __name__ == "__main__":
    pci_logic = PCIMockLogic()
    print("Segment for 'I want to cancel':", pci_logic.get_customer_segment("I want to cancel my subscription"))
    print("Suggestion for churn_risk:", pci_logic.get_suggestion("churn_risk"))
    print("Segment for 'Tell me about pricing':", pci_logic.get_customer_segment("Tell me about pricing"))
    print("Suggestion for price_sensitive:", pci_logic.get_suggestion("price_sensitive"))
    print("Segment for 'How do I upgrade?':", pci_logic.get_customer_segment("How do I upgrade?"))
    print("Suggestion for high_value_prospect:", pci_logic.get_suggestion("high_value_prospect"))