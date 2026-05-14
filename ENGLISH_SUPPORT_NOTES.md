# PROJECT UPDATES: English Language Support & Dataset Normalization

## 📝 Summary of Changes

### 1. Dataset Normalization (`normalize_datasets.py`) ✅
- ✓ Standardized column names (v1→label, v2→text)
- ✓ Converted email labels: 0→'Legitimate', 1→'Phishing'
- ✓ Removed email outliers (> 100K characters)
- ✓ Saved normalized datasets to `dataset/processed/`

**Normalized Datasets:**
- `dataset/processed/spam_normalized.csv` (5,572 SMS messages)
- `dataset/processed/phishing_email_normalized.csv` (82,474 emails)

### 2. Multilingual Preprocessing (`src/preprocessing.py`) ✅
**Added support for both Vietnamese & English:**

#### Vietnamese Patterns:
- Phone: +84, 0xxxxx
- Money: vnđ, vnd, triệu, nghìn
- Fraud keywords: trúng thưởng, xác nhận, tài khoản bị khóa, etc.
- Banks: Vietcombank, BIDV, Vietinbank, etc.

#### English Patterns:
- Phone: +1, (xxx) xxx-xxxx, 10-11 digit numbers
- Money: $, dollar, pound, £
- Fraud keywords: verify account, urgent, click here, claim prize, etc.
- Banks: Bank of America, Chase, Wells Fargo, PayPal, Amazon, etc.

#### Key Features:
- `_detect_language()`: Auto-detects language from text (Vietnamese chars detection)
- All functions accept optional `language` parameter
- Auto language detection with English as fallback
- Backward compatible - existing Vietnamese code works unchanged

### 3. Label Mapping (`src/pipeline.py`) ✅
**Updated `_coerce_labels()` to support:**
- Fraud/Phishing: spam, fraud, scam, phishing, lừa đảo
- Legitimate/Ham: ham, legit, legitimate, safe, hợp lệ

### 4. Training Scripts
- `normalize_datasets.py`: Dataset preparation
- `train_english_model.py`: Quick start for English datasets

## 🚀 Quick Start

### Step 1: Normalize Datasets
```powershell
python normalize_datasets.py
```
This creates normalized CSV files in `dataset/processed/`

### Step 2: Train Model
```powershell
python train_english_model.py
```
Or manually:
```powershell
python main.py `
  --sms-data dataset/processed/spam_normalized.csv `
  --email-data dataset/processed/phishing_email_normalized.csv `
  --output results.json
```

### Step 3: Check Results
```powershell
type results.json
```

## 📊 Dataset Statistics (After Normalization)

### SMS Dataset (spam_normalized.csv)
- Total: 5,572 messages
- Legitimate (ham): 4,825 (86.6%)
- Fraud/Spam: 747 (13.4%)
- Encoding: UTF-8
- Language: English

### Email Dataset (phishing_email_normalized.csv)
- Total: 82,474 emails (11 outliers removed)
- Legitimate: 39,589 (48.0%)
- Phishing: 42,885 (52.0%)
- Encoding: UTF-8
- Language: English

## ⚙️ How Language Detection Works

```python
from src.preprocessing import preprocess_message

# Auto-detects language
msg1 = preprocess_message("Click to verify your account")  # → English
msg2 = preprocess_message("Trúng thưởng 100 triệu đồng")   # → Vietnamese

# Or explicit language specification
msg3 = preprocess_message(text, language="en")
msg4 = preprocess_message(text, language="vi")
```

## 🔍 Key Differences: Vietnamese vs English Processing

| Feature | Vietnamese | English |
|---------|-----------|---------|
| Phone Pattern | +84, 0xxxxxxxxx | +1 (xxx) xxx-xxxx |
| Money Pattern | vnđ, triệu, nghìn | $, dollar, pound |
| Bank Keywords | Vietcombank, BIDV | Bank of America, Chase |
| Tokenization | underthesea (NLP) | Simple whitespace |
| Stopwords | Vietnamese set | English set |

## ✅ Testing

All preprocessing functions have been tested:
- ✓ Language auto-detection (correct for both languages)
- ✓ Feature extraction (URLs, phone, money, keywords detected)
- ✓ Email normalization (columns standardized, labels converted)
- ✓ SMS normalization (columns standardized, encoding fixed)

## 📝 Example: Training Pipeline Flow

```
English SMS/Email Data
    ↓
normalize_datasets.py
    ↓
Dataset standardized (cols, labels, encoding)
    ↓
main.py --sms-data ... --email-data ...
    ↓
Preprocessing (auto-detect language → English)
    ↓
Extract features (English patterns)
    ↓
Train classifier + clustering + rules
    ↓
results.json (evaluation metrics, feature importance, etc.)
```

## 🔧 Backward Compatibility

- Existing Vietnamese functionality is fully preserved
- All changes are backward compatible
- Default language is auto-detected per message
- No breaking changes to existing APIs

## 📚 Files Changed

1. `src/preprocessing.py` - Added multilingual support
2. `src/pipeline.py` - Updated label mapping for English
3. `normalize_datasets.py` - NEW: Dataset preparation script
4. `train_english_model.py` - NEW: Quick start training script

---

**Status: ✅ READY FOR TRAINING**

Both Vietnamese and English datasets are now supported. The system automatically detects the language and applies appropriate patterns for feature extraction.
