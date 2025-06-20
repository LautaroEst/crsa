
import torch

def main():
    # Example inputs
    n_meanings = 20 # Total number of meanings
    meaning_patient = 5  # The meaning we are interested in
    n_distractors = 3  # Number of distractors to select
    similarities = torch.rand(n_meanings)  # Random similarities for demonstration
    similarities = similarities / similarities.sum()  # Normalize to create a probability distribution
    indices = torch.multinomial(similarities, num_samples=n_meanings, replacement=False)
    distractors = indices[:n_distractors+1]  # Get the first n_distractors indices
    if meaning_patient not in distractors: # Ensure the selected meaning is included in the distractors
        distractors = torch.cat((distractors[:n_distractors], torch.tensor([meaning_patient])))
    distractors = distractors[torch.randperm(len(distractors))]  # Shuffle the distractors

    print("Selected Meaning:", meaning_patient)
    print("Distractors with selected meaning:", distractors.tolist())
    print("Similarities:", ["%.2f" % sim for sim in similarities[distractors].tolist()])
    print("Similarities range:", (similarities.min().item(), similarities.max().item()))




if __name__ == "__main__":
    main()