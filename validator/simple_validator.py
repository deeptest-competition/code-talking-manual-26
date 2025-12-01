from .validator import Validator
from model import TestCase
from config import get_config
import enchant
import re
from sklearn.metrics.pairwise import cosine_similarity
from llm.utils.embeddings_local import get_embedding

config = get_config()


class SimpleValidator(Validator):
    MAX_LENGTH = config["validator"]["max_request_length"]

    def __init__(self, distance_threshold: float = 0.2):
        self.storage = None  # Only needed for diversity checks

        # Multiple standard dictionaries: US + GB English
        self.dictionaries = [
            enchant.Dict("en_US"),
            enchant.Dict("en_GB"),
        ]

        self.distance_threshold = distance_threshold
        self.stored_embeddings = []

    def validate(self, test_case: TestCase) -> tuple[bool, str | None]:
        """
        Validate a TestCase's request string.
        Returns a tuple: (is_valid, reason_for_failure)
        """
        request = test_case.request
        validators = [
            self._validate_length,
            self._validate_vocabulary,
            self._validate_diversity,
        ]

        for validator in validators:
            is_valid, reason = validator(request)
            if not is_valid:
                return is_valid, reason
            
        self.stored_embeddings.append(get_embedding(request).reshape(1, -1))
        return True, None

    def _validate_length(self, request: str) -> tuple[bool, str | None]:
        tokens = request.split()
        if len(tokens) > self.MAX_LENGTH:
            return False, "Request too long"
        return True, None

    def _validate_vocabulary(self, request: str) -> tuple[bool, str | None]:
        """
        Validate that all words in the request exist in any of the configured dictionaries
        or are numeric.
        """
        valid_words = ["i'm", "i've", "we've", "12v", "gps", "dsc", "aa", "acc", "esim"]

        number_pattern = re.compile(r"^\d+(\.\d+)?$")   # integer or decimal
        percent_pattern = re.compile(r"^\d+(\.\d+)?%$") # 50%, 3.5%
        money_pattern = re.compile(r"^[\$€£]\d+(\.\d+)?$") # $100, €50.5

        for token in request.split():
            # Normalize token: lowercase and strip punctuation
            word = token.strip(".,!?;:()[]{}\"").lower()
            if not word:
                continue

            if word in valid_words:
                continue
           
            # Numbers, percentages, money
            if (number_pattern.match(word) or
                percent_pattern.match(word) or
                money_pattern.match(word)):
                continue

            # Hyphenated words
            parts = word.split("-")
            if all(part.isdigit() or
                   number_pattern.match(part) or
                   percent_pattern.match(part) or
                   money_pattern.match(part) or
                   any(d.check(part) for d in self.dictionaries) for part in parts):
                continue

            # Accept if any dictionary recognizes the word
            if not any(d.check(word) for d in self.dictionaries):
                return False, f"Non-English word detected: {token}"
        return True, None

    def _validate_diversity(self, request: str) -> tuple[bool, str | None]:
        reference_embedding = get_embedding(request).reshape(1, -1)
        
        for embedding in self.stored_embeddings:
            similarity = cosine_similarity(reference_embedding, embedding)[0][0]
            distance = (1 - similarity) / 2
            if distance < self.distance_threshold:
                return False, "The request is a duplicate of a previously validated request"
        return True, None


if __name__ == "__main__":
    validator = SimpleValidator()
    test_case = TestCase(request="This is AIS request 42", expected_warning_id="W1")

    is_valid, reason = validator.validate(test_case)
    print("Is valid:", is_valid)
    if reason:
        print("Reason:", reason)
