# Data Quality Checker (資料集有效性與品質驗證平台) 📊

[Demo](https://toydogcat.github.io/data-checker/)

`data-checker` 是一個專為機器學習、NLP 與大型語言模型（LLM）資料集設計的**自動化數據品質體檢工具庫**。提供網頁端（GitHub Pages）與 Python 工具庫（Colab / 本地端），協助開發者快速評估資料集的健康度、資訊密度、向量多樣性與 LLM 品質評分。

---

## 🌟 核心維度與評估指標

| 階段 | 評估維度 | 核心指標 | 高品質表現 |
| :--- | :--- | :--- | :--- |
| **階段一** | **統計與結構健康度** | 空值率（Missing Rate）、重複率（Duplication Rate）、亂碼破字偵測 | 空值率 &lt; 15%, 重複率 &lt; 5% |
| **階段一** | **內容資訊豐富度** | 字數長度分佈、Shannon 資訊熵 (Entropy)、詞彙多樣性 (Type-Token Ratio, TTR) | TTR &gt; 0.3, 字數適中 |
| **階段一** | **安全與合規性** | 敏感個資 (PII: 台灣身分證、電話、Email、信用卡、IP) &amp; 毒性/不當字詞偵測 | 無敏感個資與不當言論 |
| **階段二** | **Embedding 向量多樣性** | 向量空間分散度 (Dispersion)、兩兩餘弦距離 (Pairwise Cosine Distance)、PCA 主成分變異數 | 向量分佈廣泛，Topic 涵蓋全面 |
| **階段三** | **Llama / LLM as a Judge** | 抽樣 LLM 評分（流暢度 Fluency、邏輯性 Logic、資訊價值 Value 1-5 分） | 平均得分 &ge; 4.0 分 |

---

## 🌐 a 類：GitHub Pages 網頁端工具 (Phase 1 Web Checker)

可以直接在瀏覽器端載入 CSV、JSON、JSONL 或 TXT 資料，所有運算皆在客戶端完成（不傳送資料至後端）：

- 支援拖曳或上傳檔案。
- 提供一鍵載入內建測試資料集。
- 自動計算品質綜合得分（Score Ring 0 ~ 100）。
- 動態渲染欄位缺失率條狀圖與安全合規甜甜圈圖 (Chart.js)。
- 匯出完整 JSON 評估體檢報告。

### 部署說明
本專案已設定 GitHub Actions Workflow (`.github/workflows/deploy-pages.yml`)，推送至 `main` 即可自動發布為 GitHub Pages 靜態網站。

---

## 🐍 b 類：Python 工具庫 (`data_checker` Package)

支援在本地端或 Google Colab 執行。預設環境建議使用 conda `toby`。

### 1. 安裝與環境 (Conda `toby`)

```bash
conda activate toby
# 安裝依賴套件 (Phase 2 & 3 視需求安裝 optional 套件)
pip install pandas numpy scikit-learn matplotlib seaborn sentence-transformers openai
```

### 2. 使用範例 (Quick Start)

```python
import pandas as pd
from data_checker import Phase1Checker, Phase2Checker, Phase3Checker

# -----------------------------
# 階段一：結構健康度與統計體檢
# -----------------------------
df = pd.read_csv('demo/dataframe_dataset.csv')
p1 = Phase1Checker()
report1 = p1.analyze_dataframe(df)

print(f"品質綜合得分: {report1['quality_score']} / 100")
print(f"整體缺失率: {report1['overall_missing_rate']}%")
print(f"重複率: {report1['duplication_rate']}%")

# -----------------------------
# 階段二：Embedding 向量多樣性評估
# -----------------------------
texts = ["人工智慧發展迅速", "資料品質是機器學習的基石", "這服務真的靠北爛"]
p2 = Phase2Checker()
report2 = p2.evaluate_diversity(texts)

print(f"多樣性得分: {report2['diversity_score']} / 100")
print(f"評定結論: {report2['interpretation']}")

# -----------------------------
# 階段三：Llama as a Judge 抽樣審查
# -----------------------------
# 若無 API Key 可直接執行，將自動切換為內建 Heuristic 評估引擎
p3 = Phase3Checker()  
report3 = p3.audit_corpus(texts, sample_size=10)

print(f"LLM 綜合平均得分: {report3['average_scores']['overall_score']} / 5.0")
print(f"審查結果: {'✅ 通過' if report3['judge_pass'] else '⚠️ 未通過'}")
```

---

## 📁 `demo/` 資料夾說明

- `demo/dataframe_dataset.csv`：包含空值、重複列、日期格式不一、個資及毒性詞彙的測試表格數據。
- `demo/text_dataset.jsonl`：包含短對話、高品質科技文章、重複句子、敏感個資及毒性詞彙的語料庫數據。
- `demo/data_quality_checker.ipynb`：可在 Google Colab 或 Jupyter Lab 中一鍵執行的展示 Notebook。

---

## 📄 License
MIT License
