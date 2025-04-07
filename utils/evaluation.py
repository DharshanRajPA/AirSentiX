# File: Air-Senti-X/utils/evaluation.py

import random
from textblob import TextBlob

__all__ = ["get_emotion", "calculate_urgency"]

def get_emotion(text):
    """
    Estimate the emotional tone from the input text.
    
    This function first scans for keywords associated with particular emotions.
    If no keyword is found, it falls back on TextBlob's sentiment polarity.
    
    Returns:
        A string representing the estimated emotional tone ("Joy", "Anger", "Calm", etc.).
    """
    # Define basic mappings from keywords to emotions:
    keyword_emotions = {
        "Joy": ["happy", "joy", "delighted", "pleased", "cheerful", "smile"],
        "Anger": ["angry", "furious", "irate", "outraged", "annoyed"],
        "Frustration": ["frustrated", "disappointed", "upset", "sour"],
        "Hope": ["hope", "optimistic", "expectant"],
        "Calm": ["calm", "serene", "peaceful", "relaxed"]
    }
    
    text_lower = text.lower()
    
    # Check for keyword matches first
    for emotion, keywords in keyword_emotions.items():
        if any(keyword in text_lower for keyword in keywords):
            return emotion

    # Fallback: use polarity from TextBlob
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.4:
        return "Joy"
    elif polarity < -0.4:
        return "Anger"
    else:
        return "Calm"

def calculate_urgency(text):
    """
    Calculate an urgency score for the given text.
    
    The urgency score is computed based on:
      - Number of exclamation marks ('!'),
      - The count of fully uppercase words (of length > 1),
      - And the subjectivity score (from TextBlob).
    
    The score is weighted and capped to a maximum of 1.0.
    
    Returns:
        A float between 0.0 and 1.0 representing the urgency.
    """
    exclamations = text.count("!")
    upper_words = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
    
    # Use TextBlob to get subjectivity score (0.0 to 1.0)
    subjectivity = TextBlob(text).sentiment.subjectivity

    # Weighted components: adjust coefficients as desired
    raw_score = (0.1 * exclamations) + (0.05 * upper_words) + (0.3 * subjectivity)
    urgency = min(1.0, raw_score)
    
    return round(urgency, 2)

# For quick testing:
if __name__ == "__main__":
    test_text = "I am REALLY upset! This is unacceptable!!!"
    print("Test Text:", test_text)
    print("Emotion:", get_emotion(test_text))
    print("Urgency:", calculate_urgency(test_text))
