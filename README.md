# Cotiviti Intern Assessment — Atharva Shinde

**Topic 2: Clinical Decision-Making and Pattern Recognition**

AI-assisted payment integrity system that combines a classification model with an agentic reasoning layer to detect and explain potential healthcare claims fraud.

---

## What's in this repo

| Folder | Contents |
|---|---|
| `/Report` | Two-page written report (+ bibliography), MS Word format |
| `/slides` | Presentation deck summarizing the approach and results |
| `/Assessment video` | 5-minute video walkthrough of the project |
| `/code` | Jupyter notebooks — model training and the LangGraph agent |
| `/resume` | Current resume |

---

## Project summary

- **Dataset:** Synthetic healthcare claims dataset (10,000 claims, ~8.3% fraud rate)
- **Classifier:** XGBoost, trained on claim, patient, and provider features
- **Key finding:** Initial model results (94% recall, 0.997 ROC-AUC) were flagged as implausibly strong. Feature importance analysis uncovered two data leakage issues — both post-outcome fields not available at the point of a real fraud decision. Both were removed and the model retrained.
- **Final result:** 84% recall, 93% precision, 0.98 ROC-AUC — grounded in legitimate signals (claim amount, approved amount, diagnosis/procedure codes).
- **Agentic layer:** A 3-node LangGraph workflow (`predict → explain → reason`) wraps the classifier, using an LLM to generate a plain-English explanation and a recommended audit action for each claim, conditioned on the model's actual prediction.

---

## Setup (to run the code locally)

```bash
pip install -r requirements.txt   # or install xgboost, scikit-learn, langgraph, langchain-openai, python-dotenv, joblib individually
```

Create a `.env` file in the root of `/code` with:
```
OPENAI_API_KEY=your_key_here
```

Run `fraud_detection.ipynb` first to train and save the model, then `Agent.ipynb` to run the reasoning agent.

---

## Contact

Atharva Shinde
[shindeatharva0324@gmail.com](mailto:shindeatharva0324@gmail.com)
[LinkedIn](https://www.linkedin.com/in/atharva-shinde-51b646284)