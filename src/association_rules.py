from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import pandas as pd

try:
    # pyrefly: ignore [missing-import]
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    # pyrefly: ignore [missing-import]
    from mlxtend.preprocessing import TransactionEncoder
except Exception:  # pragma: no cover - optional dependency
    apriori = None
    fpgrowth = None
    association_rules = None
    TransactionEncoder = None


@dataclass
class RuleMiningConfig:
    min_support: float = 0.15
    min_confidence: float = 0.8
    min_lift: float = 1.5
    max_len: int = 3



def build_transactions(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    extra_item_columns: Sequence[str] | None = None,
    allowed_tokens: set[str] | None = None,
) -> List[List[str]]:
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    transactions: List[List[str]] = []
    extra_cols = [col for col in (extra_item_columns or []) if col in df.columns]
    
    texts = df[text_col].fillna("").astype(str).tolist()
    
    if extra_cols:
        extra_data = df[extra_cols].astype(bool).values
        for i, text in enumerate(texts):
            items = [token for token in text.split() if token]
            if allowed_tokens is not None:
                items = [token for token in items if token in allowed_tokens]
            for j, col in enumerate(extra_cols):
                if extra_data[i, j]:
                    items.append(col)
            transactions.append(sorted(set(items)))
    else:
        for text in texts:
            items = [token for token in text.split() if token]
            if allowed_tokens is not None:
                items = [token for token in items if token in allowed_tokens]
            transactions.append(sorted(set(items)))

    return transactions


def mine_association_rules(
    transactions: Sequence[Sequence[str]],
    config: RuleMiningConfig | None = None,
) -> pd.DataFrame:
    if TransactionEncoder is None or fpgrowth is None or association_rules is None:
        raise ImportError("mlxtend is required for FP-Growth rule mining")

    config = config or RuleMiningConfig()
    encoder = TransactionEncoder()
    # Try sparse output if available to save memory
    try:
        encoded = encoder.fit(transactions).transform(transactions, sparse=True)
        one_hot = pd.DataFrame.sparse.from_spmatrix(encoded, columns=encoder.columns_)
    except TypeError:
        # Fallback to dense if mlxtend version doesn't support sparse=True
        encoded = encoder.fit(transactions).transform(transactions)
        one_hot = pd.DataFrame(encoded, columns=encoder.columns_)
        
    frequent_itemsets = fpgrowth(
        one_hot,
        min_support=config.min_support,
        use_colnames=True,
        max_len=config.max_len,
    )

    if frequent_itemsets.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=config.min_confidence)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = rules[rules["lift"] >= config.min_lift].copy()
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    return rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)


def top_rules(rules: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if rules.empty:
        return rules
    columns = [column for column in ["antecedents", "consequents", "support", "confidence", "lift"] if column in rules.columns]
    return rules.loc[:, columns].head(top_n)
