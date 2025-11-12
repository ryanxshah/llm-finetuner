from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input with a chat template to SmolLM2.
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:

        #inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        #outputs = self.model.generate(**inputs)
        #return self.tokenizer.decode(outputs[0])

        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.
        """
        from tqdm import tqdm  # Importing tqdm for progress bar

        # Preventing OOM
        micro_batch_size = 16
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]

        # Focus on a single batch
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)

        do_sample = temperature > 0

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences or 1,
            eos_token_id=self.tokenizer.eos_token_id
        )

        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]

        #return self.tokenizer.batch_decode(outputs)

        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Reshape the output if needed
        if num_return_sequences is None or num_return_sequences == 1:
            return decoded
        else:
            # Group into sublists of length num_return_sequences
            return [
                decoded[i * num_return_sequences:(i + 1) * num_return_sequences]
                for i in range(len(prompts))
            ]

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # Tests if the BaseLLM is able to complete text.
    
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
