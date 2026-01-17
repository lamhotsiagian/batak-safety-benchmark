# Safety Jailbreak Benchmark Report (Judged, English)

## Overall Metrics
- Total cases: **100**
- Accuracy (refusal/behavior correctness): **0.060**
- Under-refusal count: **76**
- Over-refusal count: **0**
- Crisis-support misses: **3**
- Leakage rate (score ≥ 1): **0.780**
- Severe leakage rate (score ≥ 2): **0.000**

## By Category (lowest accuracy first)
| category              |   n |   accuracy |   leakage_ge_1 |   leakage_ge_2 |
|:----------------------|----:|-----------:|---------------:|---------------:|
| violent_wrongdoing    |  17 |   0        |       0.647059 |              0 |
| hacking               |  15 |   0        |       1        |              0 |
| fraud                 |  13 |   0        |       0.769231 |              0 |
| privacy               |  10 |   0        |       1        |              0 |
| self_harm             |   5 |   0        |       0.8      |              0 |
| weapons               |   5 |   0        |       1        |              0 |
| sexual_content_minors |   3 |   0        |       0        |              0 |
| general_safety        |   1 |   0        |       1        |              0 |
| violence              |   1 |   0        |       1        |              0 |
| hate_harassment       |   7 |   0.142857 |       0.714286 |              0 |
| illicit_drugs         |   6 |   0.166667 |       0.833333 |              0 |
| prompt_injection      |  11 |   0.181818 |       0.636364 |              0 |
| policy_evasion        |   6 |   0.333333 |       0.666667 |              0 |