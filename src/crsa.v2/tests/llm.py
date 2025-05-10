
from ..src.llm_dialog import LLM
# from transformers import AutoTokenizer

def main():
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    # input_text = "This is a test input text."
    # encoded = tokenizer(input_text, add_special_tokens=True)
    # tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    # print(tokens)
    llm = LLM.load("EleutherAI/pythia-14m", distribute=None)
    llm.distribute(devices="auto", precision="bf16-true")
    message = llm.prompt_style.apply(
        [{"role": "user", "content": "This is a test input text."},
         {"role": "assistant", "content": "This is a test output text."}]
    )
    print(message)
    # endings = llm.predict("This is a prompt", ["and this is an ending 1", "and this is an ending 2", "asdfawe"])
    # print(endings)


if __name__ == "__main__":
    main()
    