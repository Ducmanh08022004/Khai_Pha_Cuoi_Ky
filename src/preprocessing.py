from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import pandas as pd

try:
    # pyrefly: ignore [missing-import]
    from underthesea import pos_tag, word_tokenize
except Exception:  # pragma: no cover - optional dependency
    pos_tag = None
    word_tokenize = None

# ════════════════════════════════════════════════════════════════════════════════
# LANGUAGE-SPECIFIC PATTERNS AND KEYWORDS
# ════════════════════════════════════════════════════════════════════════════════

# Vietnamese patterns
VI_PHONE_REGEX = re.compile(r"(?:\+?84|0)(?:\d[ -]?){8,10}\d")
VI_MONEY_REGEX = re.compile(r"(?:\d+[\.,]?\d*)\s*(?:vnđ|vnd|đ|triệu|nghìn|k|m)\b", re.IGNORECASE)
VI_FRAUD_KEYWORDS = (
    "trúng thưởng",
    "xác nhận",
    "tài khoản bị khóa",
    "khóa tài khoản",
    "hoàn tiền",
    "nhận quà",
    "cập nhật thông tin",
    "xử lý ngay",
    "đăng nhập ngay",
    "chuyển tiền",
    "mã xác thực",
)
VI_BANK_KEYWORDS = (
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
VI_URGENT_KEEP_WORDS = {"ngay", "liền", "khẩn", "khẩn cấp", "gấp", "sớm"}
VI_STOPWORDS = {
    "và", "là", "của", "cho", "theo", "với", "một", "các", "những",
    "được", "đang", "này", "đó", "trong", "khi", "đến", "từ", "về",
    "tại", "nên", "nếu", "thì", "có", "không",
}

# English patterns
EN_PHONE_REGEX = re.compile(r"(?:\+?1|1)?(?:\s*\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}|\d{10,11})\b")
EN_MONEY_REGEX = re.compile(r"(?:\$|dollar|usd|pounds?|£)\s*(?:\d+[\.,]?\d*)|(?:\d+[\.,]?\d*)\s*(?:dollar|usd|pounds?|£)", re.IGNORECASE)
EN_FRAUD_KEYWORDS = (
    "verify account",
    "confirm identity",
    "urgent",
    "click here",
    "verify now",
    "update payment",
    "unusual activity",
    "act now",
    "limited time",
    "congratulations",
    "winner",
    "claim prize",
    "free gift",
    "otp",
    "verification code",
    "refund",
)
EN_BANK_KEYWORDS = (
    "bank of america",
    "chase",
    "wells fargo",
    "citibank",
    "bofa",
    "paypal",
    "amazon",
    "apple",
    "microsoft",
    "google",
    "bank",
)
EN_URGENT_KEEP_WORDS = {"now", "urgent", "immediately", "asap", "act", "hurry"}
EN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "will", "with", "you", "your",
}

# Language configurations
LANGUAGE_CONFIGS = {
    "vi": {
        "phone_regex": VI_PHONE_REGEX,
        "money_regex": VI_MONEY_REGEX,
        "fraud_keywords": VI_FRAUD_KEYWORDS,
        "bank_keywords": VI_BANK_KEYWORDS,
        "urgent_keep_words": VI_URGENT_KEEP_WORDS,
        "stopwords": VI_STOPWORDS,
    },
    "en": {
        "phone_regex": EN_PHONE_REGEX,
        "money_regex": EN_MONEY_REGEX,
        "fraud_keywords": EN_FRAUD_KEYWORDS,
        "bank_keywords": EN_BANK_KEYWORDS,
        "urgent_keep_words": EN_URGENT_KEEP_WORDS,
        "stopwords": EN_STOPWORDS,
    },
}

# Common patterns (both languages)
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SHORT_URL_REGEX = re.compile(r"(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|shorturl\.at|s\.id)/?\S*", re.IGNORECASE)
BANK_ACCOUNT_REGEX = re.compile(r"\b\d{8,20}\b")
CAPS_RATIO_REGEX = re.compile(r"[A-ZÀ-Ỹ]")

# Backward compatibility (default to Vietnamese)
FRAUD_KEYWORDS = VI_FRAUD_KEYWORDS
BANK_KEYWORDS = VI_BANK_KEYWORDS
URGENT_KEEP_WORDS = VI_URGENT_KEEP_WORDS
DEFAULT_STOPWORDS = VI_STOPWORDS

ENABLE_POS_TAGGING = os.environ.get("ENABLE_POS_TAGGING", "0") == "1"


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


def _detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns.
    Returns 'vi' for Vietnamese, 'en' for English.
    """
    # Vietnamese characters detection
    vietnamese_chars = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.IGNORECASE)
    
    if vietnamese_chars.search(text):
        return "vi"
    return "en"


def extract_message_features(text: object, language: str | None = None) -> Dict[str, float]:
    raw = _safe_text(text)
    lowered = raw.lower()
    
    # Auto-detect language if not provided
    if language is None:
        language = _detect_language(raw)
    
    # Get language config
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["en"])
    phone_regex = config["phone_regex"]
    money_regex = config["money_regex"]
    bank_keywords = config["bank_keywords"]
    fraud_keywords = config["fraud_keywords"]
    
    has_url = int(bool(URL_REGEX.search(raw)))
    has_shorturl = int(bool(SHORT_URL_REGEX.search(lowered)))
    has_phone = int(bool(phone_regex.search(raw)))
    has_money = int(bool(money_regex.search(lowered)))
    has_bank_account = int(bool(BANK_ACCOUNT_REGEX.search(raw)))
    bank_mention = int(any(keyword in lowered for keyword in bank_keywords))
    fraud_keyword_count = sum(lowered.count(keyword.lower()) for keyword in fraud_keywords)
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


def _tokenize(text: str, language: str | None = None) -> List[str]:
    if not text:
        return []
    
    # Try underthesea tokenization for Vietnamese
    if language == "vi" and word_tokenize is not None:
        try:
            tokenized = word_tokenize(text, format="text")
            return [token for token in tokenized.split() if token]
        except Exception:
            pass
    
    # Fallback to simple whitespace tokenization
    return [token for token in re.split(r"\s+", text) if token]


def _remove_stopwords(tokens: Sequence[str], language: str | None = None) -> List[str]:
    if language is None:
        language = "en"
    
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["en"])
    stopwords = config["stopwords"]
    urgent_keep_words = config["urgent_keep_words"]
    
    filtered_tokens: List[str] = []
    for token in tokens:
        normalized = token.strip().lower()
        if not normalized:
            continue
        if normalized in stopwords and normalized not in urgent_keep_words:
            continue
        filtered_tokens.append(normalized)
    return filtered_tokens


def normalize_text(text: object, language: str | None = None) -> str:
    raw = _safe_text(text)
    if not raw:
        return ""
    
    # Auto-detect language
    if language is None:
        language = _detect_language(raw)
    
    lowered = raw.lower()
    lowered = URL_REGEX.sub(" [URL] ", lowered)
    lowered = SHORT_URL_REGEX.sub(" [SHORTURL] ", lowered)
    
    # Use language-specific regexes
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["en"])
    lowered = config["phone_regex"].sub(" [PHONE] ", lowered)
    lowered = config["money_regex"].sub(" [MONEY] ", lowered)
    
    lowered = re.sub(r"[^0-9a-zA-ZÀ-ỹ\[\]\s_]+", " ", lowered)
    lowered = lowered.replace("_", " ")
    return _normalize_whitespace(lowered)


def preprocess_message(text: object, language: str | None = None) -> CleanedMessage:
    raw = _safe_text(text)
    
    # Auto-detect language if not provided
    if language is None:
        language = _detect_language(raw)
    
    cleaned = normalize_text(raw, language=language)
    tokens = _remove_stopwords(_tokenize(cleaned, language=language), language=language)
    pos_counts: Dict[str, int] = {}
    if ENABLE_POS_TAGGING and pos_tag is not None and cleaned:
        try:
            for _, tag in pos_tag(cleaned):
                pos_counts[tag] = pos_counts.get(tag, 0) + 1
        except Exception:
            pos_counts = {}
    return CleanedMessage(
        original=raw,
        cleaned_text=" ".join(tokens),
        tokens=tokens,
        flags={key: int(value) for key, value in extract_message_features(raw, language=language).items()},
        pos_counts=pos_counts,
    )


def preprocess_texts(texts: Iterable[object], language: str | None = None) -> List[CleanedMessage]:
    return [preprocess_message(text, language=language) for text in texts]


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text", language: str | None = None) -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    from tqdm import tqdm
    
    texts = df[text_col].tolist()
    processed_rows = []
    
    # Use tqdm for progress tracking
    for text in tqdm(texts, desc="Preprocessing messages", leave=False):
        processed_rows.append(preprocess_message(text, language=language))
    
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
