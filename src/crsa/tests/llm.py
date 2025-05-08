
from ..src.llm_dialog import LLM

def main():
    llm = LLM.load("EleutherAI/pythia-14m", distribute=None)
    llm.distribute(devices="cpu", precision="bf16-true")
    endings = llm.predict("This is a prompt", ["and this is an ending 1", "and this is an ending 2", "asdfawe"])
    print(endings)


if __name__ == "__main__":
    main()
    