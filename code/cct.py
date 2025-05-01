def load_plant_knowledge_data(filepath):
    df = pd.read_csv(filepath)
    return df.drop(columns=["InformantID"]).to_numpy()

def run_cct_model(data):
    N, M = data.shape

    with pm.Model() as cct_model:
        # Priors
        D = pm.Beta("D", alpha=2, beta=2, shape=N)  # Informant competence; favors values in (0.5, 1)
        Z = pm.Bernoulli("Z", p=0.5, shape=M)       # Consensus answers

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

# Summary
print(az.summary(trace, var_names=["D", "Z"]))

# Plots
az.plot_posterior(trace, var_names=["D"])  # competence
az.plot_posterior(trace, var_names=["Z"])  # consensus answers

def compute_majority_vote(data):
    return np.round(data.mean(axis=0)).astype(int)

cct_consensus = (trace.posterior["Z"].mean(dim=["chain", "draw"]) > 0.5).astype(int)
