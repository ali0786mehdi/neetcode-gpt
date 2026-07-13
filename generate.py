import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        output = ""

        for i in range(new_chars):
            # Crop context to the allowed context length
            context_cond = context[:, -context_length:]

            # Forward pass
            logits = model(context_cond)

            # Take logits of the last position
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Do not alter this line
            generator.set_state(initial_state)

            # Sample the next token
            next_token = torch.multinomial(probs, 1, generator=generator)

            # Append sampled token to the context
            context = torch.cat((context, next_token), dim=1)

            # Convert token to character
            output += int_to_char[next_token.item()]

        return output