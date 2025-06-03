
from ..src.speakers import LLMSpeaker

def main():
    speaker = LLMSpeaker.load(model="EleutherAI/pythia-14m")
    speaker.distribute(accelerator="auto", precision="bf16-true")

    dialog = [
        {"speaker": "patient", "content": "Hello, I have a headache."},
        {"speaker": "doctor", "content": "How long have you had this headache?"},
        {"speaker": "patient", "content": "Since yesterday evening."},
        {"speaker": "doctor", "content": "I'm afraid you have Lupus."},
    ]
    system_prompts = {
        "patient": ["You are a patient who has a headache.", "You are a patient who sometimes has Nose bleeding."],
        "doctor": ["You are a doctor."],
    }
    speakers = ["patient", "doctor"]
    speaker.get_dialog_speakers(dialog, system_prompts, speakers)


if __name__ == "__main__":
    main()