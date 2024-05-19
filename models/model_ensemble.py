import torch
from torch import nn

class ModelsEnsemble(nn.Module):
    """ ensemble model class"""
    def __init__(self, ensemble_method="confidence_weighted_average_softmax_prediction"):
        super().__init__()
        self.models = nn.ModuleList()
        ensemble_methods = {
            "average_softmax_prediction": self.average_softmax_prediction,
            "confidence_weighted_average_softmax_prediction": self.confidence_weighted_average_softmax_prediction,
            "confidence_weighted_majority_voting_prediction": self.confidence_weighted_majority_voting_prediction,
            "most_confidence_prediction": self.most_confidence_prediction
        }
        self.ensemble_method = ensemble_methods[ensemble_method]

    def append(self, model):
        self.models.append(model)

    def forward(self, x):
        img, tabular = x
        img = img.to(torch.float)
        out = self.ensemble_method((img, tabular))
        return out

    def average_softmax_prediction(self, x):
        out = self.models[0](x).softmax(dim=-1)
        for model in self.models[1:]:
            out += model(x).softmax(dim=-1)
        out = out / len(self.models)
        return out

    def confidence_weighted_average_softmax_prediction(self, x):  # weighted (by entropy) average softmax
        probs = []
        for model in self.models:
            p = model(x).softmax(dim=-1)
            p[p==0] += 10e-8
            probs.append(p)

        entropies = []
        for prob in probs:
            entropies.append(entropy(prob).view(-1, 1))
        inverse_entropies = 1 / torch.cat(entropies, dim=-1)
        weights = inverse_entropies / inverse_entropies.sum(dim=1, keepdim=True)

        out = probs[0] * weights[:, 0].view(-1, 1)
        for i in range(len(probs)-1):
            out += probs[i + 1] * weights[:, i + 1].view(-1, 1)

        return out


    def confidence_weighted_majority_voting_prediction(self, x):  # weighted (by entropy) voting
        probs = []
        for model in self.models:
            probs.append(model(x).softmax(dim=-1))

        entropies = []
        for prob in probs:
            entropies.append(entropy(prob).view(-1, 1))
        inverse_entropies = 1 / torch.cat(entropies, dim=-1)
        weights = inverse_entropies / inverse_entropies.sum(dim=1, keepdim=True)

        for i in range(len(probs)):
            probs[i] = (probs[i] == torch.max(probs[i], dim=1, keepdim=True).values).float()

        out = probs[0] * weights[:, 0].view(-1, 1)
        for i in range(len(probs)-1):
            out += probs[i + 1] * weights[:, i + 1].view(-1, 1)
        return out


    def most_confidence_prediction(self, x): # most confident decides
        probs = []
        for model in self.models:
            probs.append(model(x).softmax(dim=-1))

        entropies = []
        for prob in probs:
            entropies.append(entropy(prob).view(-1, 1))
        entropies = torch.cat(entropies, dim=-1)

        # gives weight of 1 to the most confident model and 0 to the rest
        weights = (entropies == torch.min(entropies, dim=1, keepdim=True).values).float()
        out = probs[0] * weights[:, 0].view(-1, 1)
        for i in range(len(probs)-1):
            out += probs[i + 1] * weights[:, i + 1].view(-1, 1)
        return out


def entropy(probabilities):
    negative_log_probs = -torch.log(probabilities + 1e-14)
    entropy_values = torch.sum(probabilities * negative_log_probs, dim=-1)
    return entropy_values
