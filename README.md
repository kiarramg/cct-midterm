# cct-midterm
## sample info:
### In this assignment, I implemented a Cultural Consensus Theory (CCT) model using PyMC to analyze responses from informants about local plant knowledge. The dataset consisted of binary responses (0 or 1) from multiple informants across several questions.

### For each informant’s competence, I used a Beta(2, 2) prior, which favors values between 0.5 and 1—consistent with the CCT assumption that competence is at least better than chance. For each question’s consensus answer, I used a Bernoulli(0.5) prior, reflecting no prior bias toward 0 or 1.

### Using PyMC, I defined the likelihood of informant responses based on their competence and the latent consensus answers, and I sampled from the posterior using MCMC with 2000 draws across 4 chains. I verified convergence via trace plots and az.summary(), and found no concerning diagnostics (e.g., R-hat ≈ 1).

### The model successfully estimated individual competence levels. The most competent informant had a competence score of approximately X.XX, while the least competent had a score of Y.YY. Posterior estimates of the consensus answers were derived by thresholding the mean posterior of Zj values.

### Comparing these with a naive majority vote approach, I found Z items differed between the two methods. These differences arise because the CCT model accounts for differential informant reliability, unlike the majority vote which assumes equal competence across all informants.