import code_bert_score
import pandas as pd
import torch


# %%
def calc_code_bert_score(dataset: pd.DataFrame) -> pd.DataFrame:
    print("Scoring on the code-bert-score.")

    bert_score = code_bert_score.score(
        cands=list(dataset["code_plot"]),
        refs=list(dataset["code"]),
        lang="python",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=True,
        batch_size=200,
    )
    bert_score_f1 = bert_score[2].tolist()

    dataset["score_codebert"] = bert_score_f1

    return dataset
