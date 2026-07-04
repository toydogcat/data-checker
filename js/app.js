// Data Quality Checker - Client Side JS Engine

// Regular Expressions for PII Scanner
const PII_PATTERNS = {
    taiwan_id: /[A-Za-z][1289]\d{8}/g,
    mobile_phone: /09\d{8}|0\d{1,2}-\d{6,8}/g,
    email: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
    credit_card: /\b(?:\d[ -]*?){13,16}\b/g,
    ip_address: /\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b/g
};

// Toxic / Swear Words List
const TOXIC_WORDS = [
    "幹", "三小", "靠北", "靠腰", "操你", "白痴", "智障", "蠢貨", "死全家",
    "fuck", "shit", "bitch", "bastard", "idiot", "asshole", "dumbass"
];

// Global State
let currentAnalysisReport = null;
let missingChartInstance = null;
let safetyChartInstance = null;

// Shannon Entropy Calculation
function calculateEntropy(str) {
    if (!str || str.length === 0) return 0;
    const freqs = {};
    for (let char of str) {
        freqs[char] = (freqs[char] || 0) + 1;
    }
    const len = str.length;
    let entropy = 0;
    for (let char in freqs) {
        const p = freqs[char] / len;
        entropy -= p * Math.log2(p);
    }
    return entropy;
}

// Type-Token Ratio (TTR) Calculation
function calculateTTR(texts) {
    const tokens = [];
    texts.forEach(t => {
        if (!t) return;
        const words = String(t).toLowerCase().match(/\w+/g);
        if (words) tokens.push(...words);
    });
    if (tokens.length === 0) return 0;
    const uniqueTokens = new Set(tokens);
    return uniqueTokens.size / tokens.length;
}

// Main Analysis Function for Array of Objects or Raw Strings
function analyzeDataset(data, filename = "dataset") {
    if (!data || data.length === 0) {
        alert("資料集為空或解析失敗！");
        return;
    }

    const totalRows = data.length;
    let isTabular = typeof data[0] === 'object' && data[0] !== null && !Array.isArray(data[0]);

    let columns = [];
    let columnMissingRates = {};
    let overallMissingRate = 0;
    let duplicateRows = 0;
    let duplicationRate = 0;

    let textCorpus = [];

    if (isTabular) {
        columns = Object.keys(data[0]);
        let totalCells = totalRows * columns.length;
        let missingCells = 0;

        // Duplicate row check
        const stringifiedRows = data.map(r => JSON.stringify(r));
        const uniqueSet = new Set(stringifiedRows);
        duplicateRows = totalRows - uniqueSet.size;
        duplicationRate = ((duplicateRows / totalRows) * 100).toFixed(2);

        columns.forEach(col => {
            let nullCount = 0;
            data.forEach(row => {
                const val = row[col];
                if (val === null || val === undefined || String(val).trim() === '' || String(val).toLowerCase() === 'nan') {
                    nullCount++;
                } else {
                    textCorpus.push(String(val));
                }
            });
            missingCells += nullCount;
            columnMissingRates[col] = {
                missing_count: nullCount,
                missing_rate: ((nullCount / totalRows) * 100).toFixed(2)
            };
        });

        overallMissingRate = ((missingCells / totalCells) * 100).toFixed(2);
    } else {
        // Raw text array
        textCorpus = data.map(d => String(d));
        const uniqueSet = new Set(textCorpus);
        duplicateRows = totalRows - uniqueSet.size;
        duplicationRate = ((duplicateRows / totalRows) * 100).toFixed(2);
    }

    // Text Corpus Analysis
    const textLengths = textCorpus.map(t => t.length);
    const textEntropies = textCorpus.map(t => calculateEntropy(t));
    const ttr = calculateTTR(textCorpus);

    const avgLength = textLengths.length ? (textLengths.reduce((a, b) => a + b, 0) / textLengths.length).toFixed(1) : 0;
    const avgEntropy = textEntropies.length ? (textEntropies.reduce((a, b) => a + b, 0) / textEntropies.length).toFixed(4) : 0;

    // PII Scanner
    const piiSummary = { taiwan_id: 0, mobile_phone: 0, email: 0, credit_card: 0, ip_address: 0 };
    let totalPIIFound = 0;
    let toxicRecordsCount = 0;
    let encodingIssuesCount = 0;

    const toxicRegex = new RegExp(TOXIC_WORDS.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'i');

    textCorpus.forEach(text => {
        // Encoding check
        if (text.includes('\ufffd') || text.includes('ï¿½')) {
            encodingIssuesCount++;
        }

        // PII scan
        for (let piiKey in PII_PATTERNS) {
            const matches = text.match(PII_PATTERNS[piiKey]);
            if (matches) {
                piiSummary[piiKey] += matches.length;
                totalPIIFound += matches.length;
            }
        }

        // Toxic scan
        if (toxicRegex.test(text)) {
            toxicRecordsCount++;
        }
    });

    // Score Calculation
    let score = 100.0;
    score -= Math.min(30.0, overallMissingRate * 0.8);
    score -= Math.min(25.0, duplicationRate * 1.0);
    score -= Math.min(20.0, totalPIIFound * 2.0);
    score -= Math.min(15.0, (toxicRecordsCount / (textCorpus.length || 1)) * 100 * 1.5);
    const qualityScore = Math.max(0, score.toFixed(1));

    currentAnalysisReport = {
        filename,
        totalRows,
        isTabular,
        columns,
        overallMissingRate,
        duplicateRows,
        duplicationRate,
        columnMissingRates,
        textMetrics: { avgLength, avgEntropy, ttr },
        piiSummary,
        totalPIIFound,
        toxicRecordsCount,
        encodingIssuesCount,
        qualityScore,
        checklist: [
            { item: "結構完整 (Completeness)", metric: "空值率 (Missing Rate)", value: `${overallMissingRate}%`, pass: overallMissingRate < 15 },
            { item: "乾淨程度 (Uniqueness)", metric: "重複率 (Duplication Rate)", value: `${duplicationRate}%`, pass: duplicationRate < 5 },
            { item: "資訊密度 (Density)", metric: "詞彙多樣性 TTR", value: ttr.toFixed(4), pass: ttr > 0.3 },
            { item: "合規安全 (Safety)", metric: "敏感個資 (PII Found)", value: `${totalPIIFound} 筆`, pass: totalPIIFound === 0 },
            { item: "內容毒性 (Toxicity)", metric: "不當言論筆數", value: `${toxicRecordsCount} 筆`, pass: toxicRecordsCount === 0 },
            { item: "字元健康 (Encoding)", metric: "亂碼/破字筆數", value: `${encodingIssuesCount} 筆`, pass: encodingIssuesCount === 0 }
        ]
    };

    renderReport(currentAnalysisReport);
}

// UI Render Function
function renderReport(report) {
    document.getElementById('reportSection').style.display = 'block';

    // Update Score Circle
    const scoreVal = document.getElementById('scoreValue');
    scoreVal.innerText = report.qualityScore;
    const ring = document.getElementById('scoreRing');
    ring.style.background = `conic-gradient(var(--primary) ${report.qualityScore}%, rgba(255,255,255,0.1) ${report.qualityScore}%)`;

    // Update Stats Cards
    document.getElementById('statTotalRows').innerText = report.totalRows.toLocaleString();
    document.getElementById('statMissingRate').innerText = `${report.overallMissingRate}%`;
    document.getElementById('statDupRate').innerText = `${report.duplicationRate}%`;
    document.getElementById('statPIICount').innerText = report.totalPIIFound;

    // Render Checklist Table
    const checklistBody = document.getElementById('checklistTableBody');
    checklistBody.innerHTML = report.checklist.map(row => `
        <tr>
            <td><strong>${row.item}</strong></td>
            <td>${row.metric}</td>
            <td><code>${row.value}</code></td>
            <td>
                <span class="${row.pass ? 'badge-pass' : 'badge-danger'}">
                    ${row.pass ? '✅ 合規 / 高品質' : '⚠️ 需改善'}
                </span>
            </td>
        </tr>
    `).join('');

    // Render Column Missing Rates Chart
    renderMissingChart(report.columnMissingRates);

    // Render Safety Doughnut Chart
    renderSafetyChart(report.piiSummary, report.toxicRecordsCount, report.encodingIssuesCount);
}

function renderMissingChart(colMissingData) {
    const ctx = document.getElementById('missingChart').getContext('2d');
    if (missingChartInstance) missingChartInstance.destroy();

    const labels = Object.keys(colMissingData);
    const dataVals = labels.map(k => parseFloat(colMissingData[k].missing_rate));

    if (labels.length === 0) {
        labels.push('Raw Text');
        dataVals.push(0);
    }

    missingChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: '欄位缺失率 (%)',
                data: dataVals,
                backgroundColor: 'rgba(99, 102, 241, 0.7)',
                borderColor: '#6366f1',
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true, max: 100, ticks: { color: '#94a3b8' } },
                x: { ticks: { color: '#94a3b8' } }
            },
            plugins: { legend: { labels: { color: '#f8fafc' } } }
        }
    });
}

function renderSafetyChart(piiSummary, toxicCount, encodingCount) {
    const ctx = document.getElementById('safetyChart').getContext('2d');
    if (safetyChartInstance) safetyChartInstance.destroy();

    const labels = ['身分證字號', '電話號碼', 'Email', '信用卡', 'IP位址', '毒性詞彙', '亂碼破字'];
    const dataVals = [
        piiSummary.taiwan_id,
        piiSummary.mobile_phone,
        piiSummary.email,
        piiSummary.credit_card,
        piiSummary.ip_address,
        toxicCount,
        encodingCount
    ];

    safetyChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: dataVals,
                backgroundColor: [
                    '#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6', '#ec4899', '#10b981', '#64748b'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#f8fafc' } }
            }
        }
    });
}

// File Input & Dropzone Setup
document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    // Sample Data Buttons
    document.getElementById('btnSampleCSV').addEventListener('click', loadSampleCSV);
    document.getElementById('btnSampleJSONL').addEventListener('click', loadSampleJSONL);
    const btnXml = document.getElementById('btnSampleXML');
    if (btnXml) btnXml.addEventListener('click', loadSampleXML);
    document.getElementById('btnExportReport').addEventListener('click', exportReportJSON);
});

function handleFile(file) {
    const reader = new FileReader();
    const fname = file.name;
    const ext = fname.split('.').pop().toLowerCase();

    reader.onload = (e) => {
        const content = e.target.result;
        if (ext === 'csv') {
            Papa.parse(content, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => analyzeDataset(results.data, fname)
            });
        } else if (ext === 'json') {
            try {
                const parsed = JSON.parse(content);
                const data = Array.isArray(parsed) ? parsed : [parsed];
                analyzeDataset(data, fname);
            } catch (err) { alert("JSON 格式無效！"); }
        } else if (ext === 'jsonl') {
            const lines = content.split('\n').filter(l => l.trim() !== '');
            const data = [];
            lines.forEach(l => {
                try { data.push(JSON.parse(l)); } catch (err) {}
            });
            analyzeDataset(data, fname);
        } else if (ext === 'xml') {
            parseXMLContent(content, fname);
        } else {
            // Text file
            const lines = content.split('\n').filter(l => l.trim() !== '');
            analyzeDataset(lines, fname);
        }
    };
    reader.readAsText(file);
}

// XML Parser & Converter
function parseXMLContent(xmlText, filename) {
    try {
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlText, "text/xml");
        const parserError = xmlDoc.getElementsByTagName("parsererror");
        if (parserError.length > 0) {
            alert("XML 格式解析失敗，請確認 XML 語法是否正確！");
            return;
        }

        const data = convertXMLToDataset(xmlDoc);
        if (!data || data.length === 0) {
            alert("XML 檔中未找到可評估的節點或資料列！");
            return;
        }
        analyzeDataset(data, filename);
    } catch (err) {
        alert("XML 解析發生錯誤：" + err.message);
    }
}

function convertXMLToDataset(xmlDoc) {
    const root = xmlDoc.documentElement;
    if (!root) return [];

    // Find repeated child elements
    const childTagCounts = {};
    for (let child of root.children) {
        const tag = child.tagName;
        childTagCounts[tag] = (childTagCounts[tag] || 0) + 1;
    }

    let repeatedTag = null;
    let maxFreq = 0;
    for (let tag in childTagCounts) {
        if (childTagCounts[tag] > maxFreq && childTagCounts[tag] >= 2) {
            maxFreq = childTagCounts[tag];
            repeatedTag = tag;
        }
    }

    if (repeatedTag) {
        const items = root.getElementsByTagName(repeatedTag);
        const dataset = [];
        for (let item of items) {
            const row = {};
            for (let attr of item.attributes) {
                row[`@${attr.name}`] = attr.value;
            }
            for (let child of item.children) {
                row[child.tagName] = child.textContent.trim();
            }
            if (item.children.length === 0 && item.textContent.trim()) {
                row['text'] = item.textContent.trim();
            }
            dataset.push(row);
        }
        return dataset;
    }

    const allElements = xmlDoc.getElementsByTagName('*');
    const textList = [];
    for (let el of allElements) {
        if (el.children.length === 0 && el.textContent.trim()) {
            textList.push(el.textContent.trim());
        }
    }
    return textList;
}

// Built-in Sample Datasets (Fallback for file:// protocol CORS restrictions)
const EMBEDDED_SAMPLE_CSV = `user_id,name,email,taiwan_id,phone,signup_date,notes,rating,income
1001,張小明,ming@gmail.com,A123456789,0912345678,2026-01-15,優質顧客，經常購買3C產品。,4.8,65000
1002,陳大華,hua_chen@yahoo.com,B220192837,0922334455,2026-01-16,系統操作順暢，給予好評。,5.0,82000
1003,林美麗,meili@outlook.com,H291029384,0933887766,2026/01/17,產品品質優良，物流速度快。,4.5,54000
1004,黃強生,johnson@domain.org,F192837465,0911002233,2026.01.18,無特殊備註。,3.5,
1005,李志明,,E102938475,0988776655,2026-01-19,客服人員態度很好。,4.0,48000
1001,張小明,ming@gmail.com,A123456789,0912345678,2026-01-15,優質顧客，經常購買3C產品。,4.8,65000
1006,趙敏,zhao_min@test.com,D293847561,0955443322,2026-01-20,產品包裝破損，幹這品質三小！,1.0,71000
1007,王大明,wang@demo.tw,A100200300,0912000111,2026-01-21,測試個資 A123456789 請勿外洩。,4.2,90000
1008,周杰,jay@sing.com,G182736450,,2026-01-22,,3.0,
1009,蔡依林,jolin@music.com,N293847102,0933112233,2026-01-23,這服務真的靠北爛fuck,1.5,120000
1010,許瑋甯,ann@actress.tw,F291029381,0966554433,2026-01-24,滿意這次的體驗。,4.9,98000
1011,郭台銘,terry@fox.com,A192837123,0977889900,2026-01-25,企業用戶採購數量較多。,5.0,5000000
1012,孫悟空,,J192837465,0900112233,2026-01-26,對啊,2.0,
1013,豬八戒,pig@west.org,K293847561,0911223344,2026-01-27,謝謝，很好用。,3.0,30000
1014,沙悟淨,sand@river.net,L102938475,0922334455,2026-01-28,嗯嗯,3.0,
1015,唐三藏,monk@west.org,M293847102,0933445566,2026-01-29,阿彌陀佛，慈悲喜捨。,5.0,15000
1010,許瑋甯,ann@actress.tw,F291029381,0966554433,2026-01-24,滿意這次的體驗。,4.9,98000
1016,賈碧,gabi@marley.com,P192837465,0944556677,2026-01-30,破字亂碼測試 \\ufffd ï¿½ 測試,1.0,22000`;

const EMBEDDED_SAMPLE_JSONL = `{"id": 1, "text": "人工智慧（AI）近年的快速發展，深刻地改變了全球科技產業與人類生活型態。從大型語言模型到自動駕駛技術，AI 在各個領域展現出前所未有的突破。", "category": "technology"}
{"id": 2, "text": "資料品質是機器學習與數據分析的核心基石。低品質的資料包含空值、重複值與亂碼，可能導致模型過擬合或產生錯誤結論。", "category": "data_science"}
{"id": 3, "text": "對啊", "category": "chit_chat"}
{"id": 4, "text": "謝謝", "category": "chit_chat"}
{"id": 5, "text": "嗯嗯好的收到", "category": "chit_chat"}
{"id": 6, "text": "請注意：此系統包含敏感個資，聯絡人身分證字號為 A123456789，電話為 0912345678，請務必進行去識別化處理。", "category": "privacy"}
{"id": 7, "text": "這間餐廳的服務態度真的靠北爛，餐點又貴又難吃，簡直三小！fuck", "category": "review"}
{"id": 8, "text": "台灣運輸資料易存網（TDX）提供涵蓋公車、軌道、航空、航運、自行車及停車等跨運具之綜合交通數據 API，方便開發者進行智慧交通應用整合。", "category": "transportation"}
{"id": 9, "text": "人工智慧（AI）近年的快速發展，深刻地改變了全球科技產業與人類生活型態。從大型語言模型到自動駕駛技術，AI 在各個領域展現出前所未有的突破。", "category": "technology"}
{"id": 10, "text": "自然語言處理（NLP）技術包含文本分類、情感分析、命名實體識別（NER）、機器翻譯以及問答系統等多元任務。", "category": "nlp"}`;

const EMBEDDED_SAMPLE_XML = `<?xml version="1.0" encoding="UTF-8"?>
<dataset>
    <record>
        <id>1001</id>
        <name>張小明</name>
        <email>ming@gmail.com</email>
        <taiwan_id>A123456789</taiwan_id>
        <phone>0912345678</phone>
        <signup_date>2026-01-15</signup_date>
        <notes>優質顧客，經常購買3C產品。</notes>
        <rating>4.8</rating>
    </record>
    <record>
        <id>1002</id>
        <name>陳大華</name>
        <email>hua_chen@yahoo.com</email>
        <taiwan_id>B220192837</taiwan_id>
        <phone>0922334455</phone>
        <signup_date>2026-01-16</signup_date>
        <notes>系統操作順暢，給予好評。</notes>
        <rating>5.0</rating>
    </record>
    <record>
        <id>1003</id>
        <name>李志明</name>
        <email></email>
        <taiwan_id>E102938475</taiwan_id>
        <phone>0988776655</phone>
        <signup_date>2026-01-19</signup_date>
        <notes>客服人員態度很好。</notes>
        <rating>4.0</rating>
    </record>
    <record>
        <id>1004</id>
        <name>趙敏</name>
        <email>zhao_min@test.com</email>
        <taiwan_id>D293847561</taiwan_id>
        <phone>0955443322</phone>
        <signup_date>2026-01-20</signup_date>
        <notes>產品包裝破損，幹這品質三小！</notes>
        <rating>1.0</rating>
    </record>
    <record>
        <id>1005</id>
        <name>王大明</name>
        <email>wang@demo.tw</email>
        <taiwan_id>A100200300</taiwan_id>
        <phone>0912000111</phone>
        <signup_date>2026-01-21</signup_date>
        <notes>測試個資 A123456789 請勿外洩。</notes>
        <rating>4.2</rating>
    </record>
    <record>
        <id>1001</id>
        <name>張小明</name>
        <email>ming@gmail.com</email>
        <taiwan_id>A123456789</taiwan_id>
        <phone>0912345678</phone>
        <signup_date>2026-01-15</signup_date>
        <notes>優質顧客，經常購買3C產品。</notes>
        <rating>4.8</rating>
    </record>
</dataset>`;

function loadSampleCSV() {
    fetch('demo/dataframe_dataset.csv')
        .then(res => {
            if (!res.ok) throw new Error('Fetch failed');
            return res.text();
        })
        .then(csvText => parseCSVString(csvText, "demo/dataframe_dataset.csv"))
        .catch(() => parseCSVString(EMBEDDED_SAMPLE_CSV, "demo/dataframe_dataset.csv"));
}

function parseCSVString(csvText, filename) {
    Papa.parse(csvText, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => analyzeDataset(results.data, filename)
    });
}

function loadSampleJSONL() {
    fetch('demo/text_dataset.jsonl')
        .then(res => {
            if (!res.ok) throw new Error('Fetch failed');
            return res.text();
        })
        .then(text => parseJSONLString(text, "demo/text_dataset.jsonl"))
        .catch(() => parseJSONLString(EMBEDDED_SAMPLE_JSONL, "demo/text_dataset.jsonl"));
}

function parseJSONLString(text, filename) {
    const lines = text.split('\n').filter(l => l.trim() !== '');
    const data = [];
    lines.forEach(l => {
        try { data.push(JSON.parse(l)); } catch (err) {}
    });
    analyzeDataset(data, filename);
}

function loadSampleXML() {
    fetch('demo/dataset.xml')
        .then(res => {
            if (!res.ok) throw new Error('Fetch failed');
            return res.text();
        })
        .then(xmlText => parseXMLContent(xmlText, "demo/dataset.xml"))
        .catch(() => parseXMLContent(EMBEDDED_SAMPLE_XML, "demo/dataset.xml"));
}

function exportReportJSON() {
    if (!currentAnalysisReport) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentAnalysisReport, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `data_quality_report_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}
