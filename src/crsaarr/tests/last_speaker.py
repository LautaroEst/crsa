
from ..src.llm_models.base_llm import LLM


def main():
    # model_name = "EleutherAI/pythia-70m"
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model = LLM.load(model_name)
    model.distribute("auto", precision="bf16-true")

    prompt = [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": "Hi Doctor, I am having Stuffy nose"},
        {"role": "assistant", "content": "Is it? Then do you experience Headache?"},
        {"role": "user", "content": "No, I never had anything like that."},
        {"role": "assistant", "content": "In that case, do you have any Pharynx discomfort?"},
        {"role": "user", "content": "I am experiencing that sometimes"},
        {"role": "assistant", "content": "Ok, this means you might be having"},
    ]

    print(model.prompt_style.apply(prompt))
    output = model.generate(prompt, max_new_tokens=50, temperature=0.7, return_as_token_ids=False)
    print(output)




if __name__ == "__main__":
    main()