# Discussion

## Key Takeaways
1. The project showed that sepsis in ICU patients can be modeled more effectively as a latent severity trajectory problem rather than as a static classification task.
2. The initial Gaussian HMM established feasibility, while the later Bayesian continuous-time HMM improved the framework by adding treatment-aware, severity-aware, and time-aware transitions.
3. The Bayesian model produced clinically useful outputs beyond alerts, including state probabilities, treatment-scenario comparisons, delay penalties, and patient-level decision support.

## Limtations
1. Both modeling stages are retrospective and rely only on MIMIC-IV, which limits generalizability.
2. The cohort is intentionally narrow, using a strict suspected-infection definition and restricting age to 18–65 years, which may exclude important real-world populations.
3. Neither model supports causal treatment claims; treatment effects remain observational and are vulnerable to confounding by indication, especially for vasopressors. 

## Future Work
1. Expand the cohort by testing broader suspected-infection definitions, wider age ranges, and sensitivity to alternative inclusion windows.
2. Perform external validation to assess whether the Bayesian continuous-time HMM generalizes beyond the original MIMIC-IV setting.
3. Prospectively evaluate the DSS as a complement to frontline sepsis alert systems, rather than as a replacement for them. 
