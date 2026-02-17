# 🍪 BiscuitIQ — Smart Biscuit Recommendation Website

A Flask web application that helps users find and compare biscuits based on health scores, eco scores, ingredients, and dietary preferences.

## 📦 Requirements

- Python 3.8+
- Flask (`pip install flask`)

## 🚀 Quick Start

### Option 1 — Simple run (auto-installs Flask)
```bash
python run.py
```

### Option 2 — Manual
```bash
pip install flask
python app.py
```

Then open **http://localhost:5000** in your browser.

## 🗂 File Structure

```
biscuit_app/
├── app.py          ← Main Flask application
├── run.py          ← Easy startup script
├── data.json       ← Parsed product data (462 products)
├── templates/
│   ├── base.html       ← Shared layout & styles
│   ├── index.html      ← Homepage with featured picks
│   ├── recommend.html  ← Filter + browse all products
│   ├── product.html    ← Individual product detail page
│   ├── compare.html    ← Side-by-side comparison table
│   └── about.html      ← Project info & scoring methodology
└── README.md
```

## ✨ Features

| Feature | Description |
|---|---|
| **Homepage** | Top-rated picks, quick-filter cards, instant search |
| **Explore** | Filter by brand, dietary needs, sort by health/eco/protein/sugar |
| **Product Details** | Nutrition bars, ingredient flags, grade, similar products |
| **Compare** | Side-by-side table for up to 4 products, highlights best/worst cells |
| **About** | Scoring methodology explained |

## 📊 Scoring System

**Health Score** (lower = better, range 75–170):
- Penalizes: refined flour, palm oil, artificial additives, high sugar, preservatives, trans fat, artificial colors
- Grades: A (75–90), B (91–110), C (111–130), D (131–150), F (151+)

**Eco Score** (higher = better, range 70–90):
- Rewards: recyclable packaging, eco labels, vegan, organic, local manufacturing

## 🔍 Filters Available

- Search by name or brand
- Filter by brand
- Dietary: Vegan, Organic, Cruelty-Free
- Avoid: Trans Fat, Artificial Colors, Preservatives, Added Sugar, Artificial Flavours
- Sort: Health Score, Eco Score, Protein, Low Sugar, Low Fat, Name A–Z

## Data

462 products from 80+ Indian biscuit brands including Britannia, Parle, Sunfeast, Oreo, ITC, KitKat, Cadbury and more.
