# Safety Jailbreak Benchmark Report (Judged, English)

## Overall Metrics
- Total cases: **100**
- Accuracy (refusal/behavior correctness): **0.680**
- Under-refusal count: **19**
- Over-refusal count: **2**
- Crisis-support misses: **3**
- Leakage rate (score ≥ 1): **0.150**
- Severe leakage rate (score ≥ 2): **0.000**

## By Category (lowest accuracy first)
| category              |   n |   accuracy |   leakage_ge_1 |   leakage_ge_2 |
|:----------------------|----:|-----------:|---------------:|---------------:|
| self_harm             |   5 |   0        |       0        |              0 |
| general_safety        |   1 |   0        |       1        |              0 |
| violence              |   1 |   0        |       1        |              0 |
| policy_evasion        |   6 |   0.5      |       0.166667 |              0 |
| prompt_injection      |  11 |   0.636364 |       0        |              0 |
| illicit_drugs         |   6 |   0.666667 |       0.333333 |              0 |
| fraud                 |  13 |   0.692308 |       0.153846 |              0 |
| privacy               |  10 |   0.7      |       0.3      |              0 |
| hacking               |  15 |   0.733333 |       0.133333 |              0 |
| weapons               |   5 |   0.8      |       0.2      |              0 |
| violent_wrongdoing    |  17 |   0.823529 |       0.117647 |              0 |
| hate_harassment       |   7 |   0.857143 |       0        |              0 |
| sexual_content_minors |   3 |   1        |       0        |              0 |