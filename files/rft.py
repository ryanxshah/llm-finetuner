from .base_llm import BaseLLM
from .sft import test_model, TokenizedDataset
from .data import Dataset
from peft import get_peft_model, LoraConfig
from peft.utils.peft_types import TaskType
from transformers import TrainingArguments, Trainer


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def format_example(question: str, target: float, reasoning: str):
    return {
        "question": question,
        "answer": reasoning
    }


def train_model(
    output_dir: str,
    **kwargs,
):

    base_llm = BaseLLM()
    tokenizer = base_llm.tokenizer
    model = base_llm.model

    config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=4*8 #4 or 5 * r
    )

    model = get_peft_model(model, config)
    model.enable_input_require_grads()

    rftset = Dataset("rft")
    tokenized_dataset = TokenizedDataset(tokenizer, rftset, format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=5,
        gradient_checkpointing=True,
        learning_rate=5e-4,
        logging_dir=output_dir,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
