"""
fuzzy_engine.t5_model
=====================
T5 Neural Network address correction model.

This is the core ML brain of the enterprise system.
The T5 model has been fine-tuned on address data to understand
the structure and patterns of Indian addresses. Given a garbled
input, it generates the correct, properly formatted address.

Usage:
    model = T5AddressModel("models/t5_address")
    corrected = model.correct("prestig apertmnebt bangalor")
    # -> "prestige apartment bangalore"
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5AddressModel:
    """
    Fine-tuned T5 model for address correction.

    The model takes garbled/misspelled address text and generates
    the corrected version using learned patterns from training data.
    """

    def __init__(self, model_path: str = "models/t5_address"):
        """
        Load the fine-tuned T5 model and tokenizer.

        Args:
            model_path: Path to the saved T5 model directory.
        """
        print("  Loading T5 ML model...")
        self._tokenizer = T5Tokenizer.from_pretrained(model_path)
        self._model = T5ForConditionalGeneration.from_pretrained(model_path)
        self._model.eval()

        # Use GPU if available, otherwise CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        print(f"  T5 model loaded on {self._device}")

    def correct(self, raw_address: str, num_beams: int = 4,
                max_length: int = 96) -> str:
        """
        Use the T5 model to correct a garbled address.

        Args:
            raw_address: The raw/garbled address string.
            num_beams:   Beam search width (higher = better but slower).
            max_length:  Maximum output token length.

        Returns:
            The model's best corrected address string.
        """
        # Prefix that the model was trained with
        input_text = f"correct address: {raw_address.lower().strip()}"

        # Tokenize
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(self._device)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        # Decode
        corrected = self._tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()

        return corrected

    def correct_batch(self, addresses: list, num_beams: int = 4,
                      max_length: int = 96, batch_size: int = 16) -> list:
        """
        Correct multiple addresses in batches (for bulk processing).

        Args:
            addresses:  List of raw address strings.
            num_beams:  Beam search width.
            max_length: Maximum output token length.
            batch_size: Number of addresses per batch.

        Returns:
            List of corrected address strings.
        """
        results = []
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i + batch_size]
            input_texts = [f"correct address: {a.lower().strip()}" for a in batch]

            inputs = self._tokenizer(
                input_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

            for out in outputs:
                decoded = self._tokenizer.decode(
                    out, skip_special_tokens=True
                ).strip()
                results.append(decoded)

        return results
