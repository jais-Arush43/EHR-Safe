import torch

class StochasticNormalizer:
    def __init__(self):
        self.params = {}

    def stochastic_normalize(self, X: torch.Tensor, key=None):
        X = X.float()
        unique_vals, counts = torch.unique(X, return_counts=True)
        N = X.numel()
        X_hat = torch.empty_like(X, dtype=torch.float32, device=X.device)
        lower_bound = 0.0
        params = {}

        for val, count in zip(unique_vals, counts):
            ratio = count.item() / N
            upper_bound = lower_bound + ratio
            mask = X == val
            X_hat[mask] = torch.rand(mask.sum(), device=X.device) * (upper_bound - lower_bound) + lower_bound
            params[val.item()] = [lower_bound, upper_bound]
            lower_bound = upper_bound

        if key is not None:
            self.params[key] = params
        return X_hat

    def stochastic_renormalize(self, X_hat: torch.Tensor, key=None):
        X_hat = X_hat.float()
        X = torch.zeros_like(X_hat, dtype=torch.float32, device=X_hat.device)
        if key is not None:
            params = self.params[key]

        for val, (low, high) in params.items():
            mask = (X_hat >= low) & (X_hat < high)
            X[mask] = val

        # Handle edge case where X_hat == 1.0
        mask = X_hat == 1.0
        for val, (low, high) in params.items():
            if abs(high - 1.0) < 1e-8:
                X[mask] = val
        return X

    def normalize_sample(self, X: torch.Tensor, key):
        if key not in self.params:
            raise ValueError(f"No parameters found for key '{key}'. Load or train first.")

        params = self.params[key]
        X_hat = torch.zeros_like(X, dtype=torch.float32, device=X.device)
        for val, (low, high) in params.items():
            mask = X == val
            if mask.any():
                X_hat[mask] = torch.rand(mask.sum(), device=X.device) * (high - low) + low
        return X_hat

    def save_params(self, filepath):
        torch.save(self.params, filepath)

    def load_params(self, filepath):
        self.params = torch.load(filepath)
