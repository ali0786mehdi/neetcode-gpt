import torch
import torch.nn.functional as F

class Solution:
    def train(self, model, data, epochs, context_length, batch_size, lr):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        loss = None

        for epoch in range(epochs):
            torch.manual_seed(epoch)

            idx = torch.randint(
                0,
                len(data) - context_length,
                (batch_size,)
            )

            X = torch.stack([data[i:i + context_length] for i in idx])
            Y = torch.stack([data[i + 1:i + context_length + 1] for i in idx])

            logits = model(X)

            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            Y = Y.reshape(B * T)

            loss = F.cross_entropy(logits, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)