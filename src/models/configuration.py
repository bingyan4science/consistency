from transformers import PretrainedConfig


class Config(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        #tokenizer_name='gpt2',
        tokenizer_name=None,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name if tokenizer_name is not None else base_model
        super().__init__(**kwargs)
