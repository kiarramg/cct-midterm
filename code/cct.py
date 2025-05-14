import pandas as pd
import pymc as pm
import arviz as az
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend that doesn't open windows
import matplotlib.pyplot as plt
import arviz as az

def load_plant_knowledge_data(filepath):
    df = pd.read_csv(filepath)
    return df.drop(columns=["Informant"]).to_numpy()


data = load_plant_knowledge_data("../data/plant_knowledge.csv")


def run_cct_model(data):
    N, M = data.shape

    with pm.Model() as cct_model:
        # Priors
        D = pm.Beta("D", alpha=2, beta=2, shape=N)  # Informant competence; favors values in (0.5, 1)
        Z = pm.Bernoulli("Z", p=0.7, shape=M)       # Consensus answers

        # Reshape for broadcasting
        D_reshaped = D[:, None]  # shape (N, 1)
        Z_reshaped = Z[None, :]  # shape (1, M)

        # Probabilities
        p = Z_reshaped * D_reshaped + (1 - Z_reshaped) * (1 - D_reshaped)

        # Likelihood
        X = pm.Bernoulli("X", p=p, observed=data)

        # Sampling
        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, return_inferencedata=True)

    return trace

trace = run_cct_model(data)

# Show convergence diagnostics plot (silently generate, then discard)
az.plot_trace(trace, var_names=["D", "Z"])
plt.tight_layout()
plt.close()

# Summary
print(az.summary(trace, var_names=["D", "Z"]))

# Plot posterior of informant competence (D) (silently)
az.plot_posterior(trace, var_names=["D"], coords={"D_dim_0": list(range(5))})
plt.close()

# Plot posterior of consensus answers (Z) (silently)
az.plot_posterior(trace, var_names=["Z"], coords={"Z_dim_0": list(range(5))})
plt.close()


def compute_majority_vote(data):
    return np.round(data.mean(axis=0)).astype(int)

cct_consensus = (trace.posterior["Z"].mean(dim=["chain", "draw"]) > 0.5).astype(int)

# CHAT 5/13: Identify most and least competent informants
competence_means = trace.posterior["D"].mean(dim=["chain", "draw"]).values
most_competent = np.argmax(competence_means)
least_competent = np.argmin(competence_means)
print(f"Most competent: Informant {most_competent}, Score: {competence_means[most_competent]:.2f}")
print(f"Least competent: Informant {least_competent}, Score: {competence_means[least_competent]:.2f}")

# CHAT 5/13: Compare CCT consensus with majority vote
majority_vote = compute_majority_vote(data)
differences = (cct_consensus.values != majority_vote).sum()
print(f"Differences between CCT model and majority vote: {differences}")

# CHAT 5/13: print("\n--- Summary ---")
print("The CCT model estimated informant competence and consensus answers using Bayesian inference.")
print("We found the most competent informant was", most_competent, "with a competence of", round(competence_means[most_competent], 2))
print("The consensus answer key differed from the naive majority vote on", differences, "out of", data.shape[1], "items.")