from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd

try:
    from underthesea import pos_tag, word_tokenize
except Exception:  # pragma: no cover - optional dependency
    pos_tag = None
    word_tokenize = None

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SHORT_URL_REGEX = re.compile(r"(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|shorturl\.at|s\.id)/?\S*", re.IGNORECASE)
PHONE_REGEX = re.compile(r"(?:\+?84|0)(?:\d[ -]?){8,10}\d")
MONEY_REGEX = re.compile(r"(?:\d+[\.,]?\d*)\s*(?:vnđ|vnd|đ|dollar|usd|triệu|nghìn|k|m)", re.IGNORECASE)
BANK_ACCOUNT_REGEX = re.compile(r"\b\d{8,20}\b")
CAPS_RATIO_REGEX = re.compile(r"[A-ZÀ-Ỹ]")

FRAUD_KEYWORDS = (
    "trúng thưởng",
    "xác nhận",
    "tài khoản bị khóa",
    "khóa tài khoản",
    "click",
    "urgent",
    "free",
    "hoàn tiền",
    "nhận quà",
    "cập nhật thông tin",
    "xử lý ngay",
    "đăng nhập ngay",
    "chuyển tiền",
    "OTP",
    "mã xác thực",
)
BANK_KEYWORDS = (
    "vietcombank",
    "bidv",
    "vietinbank",
    "agribank",
    "mbbank",
    "mb bank",
    "techcombank",
    "acb",
    "sacombank",
    "shb",
    "tpbank",
    "vpbank",
    "ncb",
    "eximbank",
    "bank",
)
URGENT_KEEP_WORDS = {"ngay", "liền", "khẩn", "khẩn cấp", "gấp", "sớm"}
DEFAULT_STOPWORDS = {
    "và",
    "là",
    "của",
    "cho",
    "theo",
    "với",
    "một",
    "các",
    "những",
    "được",
    "đang",
    "này",
    "đó",
    "trong",
    "khi",
    "đến",
    "từ",
    "về",
    "tại",
    "nên",
    "nếu",
    "thì",
    "có",
    "không",
}


@dataclass
class CleanedMessage:
    original: str
    cleaned_text: str
    tokens: List[str]
    flags: Dict[str, int]
    pos_counts: Dict[str, int]


def _safe_text(text: object) -> str:
    if text is None:
        return ""
    if pd.isna(text):
        return ""
    return str(text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_message_features(text: object) -> Dict[str, float]:
    raw = _safe_text(text)
    lowered = raw.lower()
    has_url = int(bool(URL_REGEX.search(raw)))
    has_shorturl = int(bool(SHORT_URL_REGEX.search(lowered)))
    has_phone = int(bool(PHONE_REGEX.search(raw)))
    has_money = int(bool(MONEY_REGEX.search(lowered)))
    has_bank_account = int(bool(BANK_ACCOUNT_REGEX.search(raw)))
    bank_mention = int(any(keyword in lowered for keyword in BANK_KEYWORDS))
    fraud_keyword_count = sum(lowered.count(keyword.lower()) for keyword in FRAUD_KEYWORDS)
    exclamation_count = raw.count("!")
    caps_letters = len(CAPS_RATIO_REGEX.findall(raw))
    alpha_letters = sum(1 for char in raw if char.isalpha())
    capslock_ratio = float(caps_letters / alpha_letters) if alpha_letters else 0.0
    msg_length = len(raw.split())
    return {
        "has_url": has_url,
        "has_shorturl": has_shorturl,
        "has_phone": has_phone,
        "has_money": has_money,
        "has_bank_account": has_bank_account,
        "bank_mention": bank_mention,
        "fraud_keyword_count": float(fraud_keyword_count),
        "exclamation_count": float(exclamation_count),
        "capslock_ratio": capslock_ratio,
        "msg_length": float(msg_length),
    }


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    if word_tokenize is not None:
        try:
            tokenized = word_tokenize(text, format="text")
            return [token for token in tokenized.split() if token]
        except Exception:
            pass
    return [token for token in re.split(r"\s+", text) if token]


def _remove_stopwords(tokens: Sequence[str]) -> List[str]:
    filtered_tokens: List[str] = []
    for token in tokens:
        normalized = token.strip().lower()
        if not normalized:
            continue
        if normalized in DEFAULT_STOPWORDS and normalized not in URGENT_KEEP_WORDS:
            continue
        filtered_tokens.append(normalized)
    return filtered_tokens


def normalize_text(text: object) -> str:
    raw = _safe_text(text)
    if not raw:
        return ""
    lowered = raw.lower()
    lowered = URL_REGEX.sub(" [URL] ", lowered)
    lowered = SHORT_URL_REGEX.sub(" [SHORTURL] ", lowered)
    lowered = PHONE_REGEX.sub(" [PHONE] ", lowered)
    lowered = MONEY_REGEX.sub(" [MONEY] ", lowered)
    lowered = re.sub(r"[^0-9a-zA-ZÀ-ỹ\[\]\s_]+", " ", lowered)
    lowered = lowered.replace("_", " ")
    return _normalize_whitespace(lowered)


def preprocess_message(text: object) -> CleanedMessage:
    raw = _safe_text(text)
    cleaned = normalize_text(raw)
    tokens = _remove_stopwords(_tokenize(cleaned))
    pos_counts: Dict[str, int] = {}
    if pos_tag is not None and cleaned:
        try:
            for _, tag in pos_tag(cleaned):
                pos_counts[tag] = pos_counts.get(tag, 0) + 1
        except Exception:
            pos_counts = {}
    return CleanedMessage(
        original=raw,
        cleaned_text=" ".join(tokens),
        tokens=tokens,
        flags={key: int(value) for key, value in extract_message_features(raw).items()},
        pos_counts=pos_counts,
    )


def preprocess_texts(texts: Iterable[object]) -> List[CleanedMessage]:
    return [preprocess_message(text) for text in texts]


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    processed_rows = [preprocess_message(text) for text in df[text_col].tolist()]
    output = df.copy()
    output["clean_text"] = [row.cleaned_text for row in processed_rows]
    output["token_count"] = [len(row.tokens) for row in processed_rows]

    feature_frame = pd.DataFrame([row.flags for row in processed_rows])
    for column in feature_frame.columns:
        output[column] = feature_frame[column].astype(float)

    if any(row.pos_counts for row in processed_rows):
        pos_frame = pd.DataFrame([row.pos_counts for row in processed_rows]).fillna(0)
        for column in pos_frame.columns:
            output[f"pos_{column.lower()}"] = pos_frame[column].astype(float)

    return output
