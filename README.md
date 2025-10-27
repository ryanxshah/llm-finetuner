# LLM Finetuner - A Reusable Pipeline for Fine-Tuning LLMs

In this project, I developed a specialized unit conversion chatbot by fine-tuning Hugging Face’s SmolLM2 model. Aside from the product itself, another outcome is a reusable pipeline for dataset resampling and model training, which leverages chain-of-thought prompting, supervised fine-tuning, and rejection sampling to enhance accuracy.

## Chat with the model!

You can chat with the model [here](https://github.com/ryanxshah/self-driving-car/tree/main/videos).

Try asking a question such as "What is 3.2km in miles?".
Note that the model will respond very poorly if asked any other types of questions. This is due to the fact that the base model is very small and struggles with generic language tasks. The tradeoff made in this project was to fine-tune this small model in such a way that it performs very well on the specialized task of unit conversion, even if it doesn't work well for general purposes.
