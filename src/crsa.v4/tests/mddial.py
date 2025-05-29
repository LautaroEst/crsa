
from ..src.datasets import MDDialDataset
from ..src.llm_models.prompts import model_name_to_prompt_style

def main():
    prompt_style = model_name_to_prompt_style("Llama3.2-1B-Instruct")
    dataset = MDDialDataset(prompt_style, "train")
    # print(dataset._create_patient_system_prompt(["fever", "cough"]))
    print(dataset._create_doctor_system_prompt())
        


if __name__ == "__main__":
    main()