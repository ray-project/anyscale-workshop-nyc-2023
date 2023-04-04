import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import transformers
from ray.train.predictor import Predictor
from transformers import AutoTokenizer

transformers.set_seed(42)
warnings.simplefilter("ignore")


class HuggingFaceModelPredictor(Predictor):
    """
    A Ray Predictor for Hugging Face models that generates text given input data.

    Args:
        model (transformers.PreTrainedModel): A trained Hugging Face model.
        tokenizer (Optional[transformers.PreTrainedTokenizerBase]): A tokenizer
        that can tokenize input text.
        preprocessor (Optional[Callable]): A function that takes raw input data
        and returns tokenized input data.
        use_gpu (bool): Whether to use a GPU or CPU for prediction.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        use_gpu: bool = False,
    ) -> None:
        super().__init__(preprocessor)
        self.model = model
        self.use_gpu = use_gpu
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Any,
        model_cls: Any,
        *,
        tokenizer: Optional[Any] = None,
        use_gpu: bool = False,
        **get_model_kwargs: Any,
    ) -> "HuggingFaceModelPredictor":
        """
        Create a HuggingFaceModelPredictor from a checkpoint.

        Args:
            checkpoint (Any): A checkpoint containing a trained Hugging Face model.
            model_cls (Any): The type of Hugging Face model to load from the checkpoint.
            tokenizer (Optional[Any]): A tokenizer that can tokenize input text.
            use_gpu (bool): Whether to use a GPU or CPU for prediction.
            **get_model_kwargs (Any): Additional keyword arguments for loading
            the Hugging Face model.

        Returns:
            HuggingFaceModelPredictor: A Ray Predictor for the Hugging Face model.
        """
        if not tokenizer:
            tokenizer = AutoTokenizer
        if isinstance(tokenizer, type):
            tokenizer = checkpoint.get_tokenizer(tokenizer)
        return cls(
            checkpoint.get_model(model_cls, **get_model_kwargs),
            tokenizer=tokenizer,
            preprocessor=checkpoint.get_preprocessor(),
            use_gpu=use_gpu,
        )

    def _predict_numpy(
        self,
        data: Dict[str, Any],
        feature_columns: Optional[List[str]] = None,
        **generate_kwargs: Any,
    ) -> pd.DataFrame:
        """
        Generates text given input data.

        Args:
            data (Dict[str, Any]): A dictionary of input data.
            feature_columns (Optional[List[str]]): A list of feature column names
            to use for prediction.
            **generate_kwargs (Any): Additional keyword arguments for generating text.

        Returns:
            pd.DataFrame: A Pandas DataFrame with a single column "generated_output"
            containing the generated text.
        """
        # we get already tokenized text here because we have the tokenizer as an AIR preprocessor
        if feature_columns:
            data = {k: v for k, v in data.items() if k in feature_columns}

        data = {
            k: torch.from_numpy(v).to(device=self.model.device) for k, v in data.items()
        }
        generate_kwargs = {**data, **generate_kwargs}

        outputs = self.model.generate(**generate_kwargs)
        return pd.DataFrame(
            self.tokenizer.batch_decode(outputs, skip_special_tokens=True),
            columns=["generated_output"],
        )
