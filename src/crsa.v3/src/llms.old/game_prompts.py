

from .prompts import Llama3, Pythia


class MutualFriendsInstructions:

    instructions = (
        "You have to find which frinds have in common with the user.\n"
        "Here is a list of your frinds:\n"
        "{meaning}"
    )


class Llama3MutualFriends(Llama3, MutualFriendsInstructions):

    def apply(self, messages, meaning):
        instructions_with_meaning = self.instructions.format(meaning=meaning)
        messages = [
            {"role": "system", "content": instructions_with_meaning}
        ] + messages
        return super().apply(messages)


