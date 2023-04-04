from typing import Any, Dict

from transformers import T5Tokenizer


def preprocess_function(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tokenizes the input and instruction pairs in a batch using the T5 tokenizer
    from the Google/flan-t5-base model, and returns a dictionary containing the
    encoded inputs and labels.

    Args:
        batch: A dictionary containing at least two keys, "instruction" and
        "input", whose values are lists of strings.

    Returns:
        A dictionary containing the encoded inputs and labels, as returned by
        the T5 tokenizer.
    """
    model_name = "google/flan-t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    encoded_inputs = tokenizer(
        list(batch["instruction"]),
        list(batch["input"]),
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    encoded_inputs["labels"] = encoded_inputs["input_ids"].copy()

    return dict(encoded_inputs)
