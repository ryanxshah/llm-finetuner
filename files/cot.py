from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template.
        """

        messages = [
            {"role": "system", "content": "You are a helpful assistant that performs unit conversions. Be concise and show reasoning."},
            
            {"role": "user", "content": "How does 5 kg measure up in terms of gram?"},
            {"role": "assistant", "content": "1 kg = 1000 grams. 5 * 1000 = <answer>5000.0</answer>"},

            {"role": "user", "content": "How many in are there per 8 ft?"},
            {"role": "assistant", "content": "1 ft = 12 in. 8 * 12 = <answer>96.0</answer>"},

            {"role": "user", "content": "How many cm is 2 ft?"},
            {"role": "assistant", "content": "1 ft = 30.48 cm. 2 * 30.48 = <answer>60.96</answer>"},

            {"role": "user", "content": question}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
            )
    
        return prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
