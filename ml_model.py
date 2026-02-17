"""
BiscuitIQ — ML Recommendation Engine
Algorithm: Content-Based Filtering using Cosine Similarity

Feature Engineering:
  1. Numerical features  — nutrition (energy, fat, protein, carbs, sugars),
                           binary flags (added_sugar, preservatives, vegan, etc.),
                           health_score, eco_score
                           → normalized with MinMaxScaler (weight: 60%)

  2. Text features       — TF-IDF on ingredient text (top 100 terms)
                           → captures ingredient overlap between products (weight: 40%)

  3. Combined matrix     → cosine similarity computed across all 462 products

Usage:
  from ml_model import get_similar, get_cluster_label, get_recommendations_for_profile
"""

import json, os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# ── Load data ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, 'data.json'), 'r') as f:
    PRODUCTS = json.load(f)

# ── Feature columns ──────────────────────────────────────────────────────────
NUM_FEATURES = [
    'energy', 'fat', 'protein', 'carbs', 'sugars',
    'added_sugar', 'added_salt', 'preservatives',
    'artificial_flavours', 'artificial_colors', 'trans_fat',
    'organic', 'vegan', 'cruelty_free',
    'health_score_raw', 'eco_score_raw'
]

# ── Build feature matrix ─────────────────────────────────────────────────────
def _build_features():
    # 1. Numerical
    X_num = np.array(
        [[p.get(f, 0) or 0 for f in NUM_FEATURES] for p in PRODUCTS],
        dtype=float
    )
    scaler = MinMaxScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # 2. TF-IDF on ingredients
    ingredients = [p.get('ingredients', '') or '' for p in PRODUCTS]
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    X_text = tfidf.fit_transform(ingredients).toarray()

    # 3. Combined (60% numerical, 40% text)
    X_combined = np.hstack([X_num_scaled * 0.6, X_text * 0.4])

    # 4. Cosine similarity matrix
    sim_matrix = cosine_similarity(X_combined)

    # 5. KMeans clustering (5 health clusters)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_num_scaled)

    return sim_matrix, clusters, X_num_scaled, scaler, tfidf, X_combined

# Build on import
SIM_MATRIX, CLUSTERS, X_NUM_SCALED, SCALER, TFIDF, X_COMBINED = _build_features()

# Map product id -> index
ID_TO_IDX = {str(p['id']): i for i, p in enumerate(PRODUCTS)}

# Cluster labels based on avg health score per cluster
_cluster_health = {}
for i, p in enumerate(PRODUCTS):
    c = int(CLUSTERS[i])
    if c not in _cluster_health:
        _cluster_health[c] = []
    _cluster_health[c].append(p['health_score_raw'])

_cluster_avg = {c: np.mean(v) for c, v in _cluster_health.items()}
# Higher avg health score = healthier, so sort DESCENDING
_sorted_clusters = sorted(_cluster_avg, key=lambda c: _cluster_avg[c], reverse=True)

# Map cluster id → human label (rank 0 = highest score = Healthiest)
CLUSTER_LABELS = {}
labels = ['Healthiest', 'Healthy', 'Moderate', 'Indulgent', 'Most Indulgent']
for rank, cid in enumerate(_sorted_clusters):
    CLUSTER_LABELS[cid] = labels[rank]

CLUSTER_COLORS = {
    'Healthiest':      '#DCFCE7',
    'Healthy':         '#D1FAE5',
    'Moderate':        '#FEF9C3',
    'Indulgent':       '#FED7AA',
    'Most Indulgent':  '#FEE2E2',
}
CLUSTER_TEXT_COLORS = {
    'Healthiest':      '#166534',
    'Healthy':         '#065F46',
    'Moderate':        '#92400E',
    'Indulgent':       '#9A3412',
    'Most Indulgent':  '#991B1B',
}

# ── Public API ───────────────────────────────────────────────────────────────

def get_similar(product_id: str, top_n: int = 6) -> list:
    """
    Return top_n most similar products to the given product_id
    using cosine similarity on combined feature matrix.
    """
    idx = ID_TO_IDX.get(str(product_id))
    if idx is None:
        return []

    scores = list(enumerate(SIM_MATRIX[idx]))
    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, score in scores:
        if i == idx:
            continue
        p = dict(PRODUCTS[i])
        p['similarity'] = round(float(score) * 100, 1)  # as percentage
        p['cluster_label'] = CLUSTER_LABELS[int(CLUSTERS[i])]
        results.append(p)
        if len(results) >= top_n:
            break
    return results


def get_cluster_label(product_id: str) -> dict:
    """Return the cluster label and style for a given product."""
    idx = ID_TO_IDX.get(str(product_id))
    if idx is None:
        return {'label': 'Unknown', 'bg': '#F3F4F6', 'color': '#6B7280'}
    label = CLUSTER_LABELS[int(CLUSTERS[idx])]
    return {
        'label': label,
        'bg': CLUSTER_COLORS[label],
        'color': CLUSTER_TEXT_COLORS[label],
    }


def get_recommendations_for_profile(preferences: dict, top_n: int = 12) -> list:
    """
    Content-based recommendation from a user preference profile.
    Higher health_score_raw = better (grade A >= 150, F < 90).

    health_preference:
      'healthy'  -> only A/B grade products (score >= 130), sorted by highest score
      'moderate' -> balanced: B/C grade (score 110-149) + good eco score
      'any'      -> all products, sorted by health score descending
    """
    health_pref = preferences.get('health_preference', 'any')

    # ── Step 1: Hard filter by health_preference grade ──────────────────────
    def passes_health_filter(p):
        s = p.get('health_score_raw', 0)
        if health_pref == 'healthy':
            return s >= 130          # Only A and B grade
        elif health_pref == 'moderate':
            return 100 <= s < 150    # B and C grade — balanced
        return s > 0                 # any — exclude 0/missing

    # ── Step 2: Hard filter by dietary preferences ───────────────────────────
    def passes_dietary_filter(p):
        if preferences.get('no_preservatives') and p.get('preservatives'):
            return False
        if preferences.get('no_artificial_colors') and p.get('artificial_colors'):
            return False
        if preferences.get('no_trans_fat') and p.get('trans_fat'):
            return False
        if preferences.get('vegan') and not p.get('vegan'):
            return False
        if preferences.get('organic') and not p.get('organic'):
            return False
        return True

    # ── Step 3: Nutrition slider filters ─────────────────────────────────────
    def passes_nutrition_filter(p):
        if p.get('sugars', 0) > preferences.get('max_sugar', 50):
            return False
        if p.get('fat', 0) > preferences.get('max_fat', 35):
            return False
        if p.get('protein', 0) < preferences.get('min_protein', 0):
            return False
        return True

    # ── Step 4: Filter products ───────────────────────────────────────────────
    filtered = []
    for i, p in enumerate(PRODUCTS):
        if not passes_health_filter(p):
            continue
        if not passes_dietary_filter(p):
            continue
        if not passes_nutrition_filter(p):
            continue
        pd = dict(p)
        pd['cluster_label'] = CLUSTER_LABELS[int(CLUSTERS[i])]
        filtered.append((i, pd))

    # ── Step 5: Score using cosine similarity to an ideal vector ─────────────
    feat_idx = {f: i for i, f in enumerate(NUM_FEATURES)}
    ideal = np.zeros(len(NUM_FEATURES))

    # Set ideal health/eco scores based on preference
    if health_pref == 'healthy':
        ideal[feat_idx['health_score_raw']] = 170   # max possible
        ideal[feat_idx['eco_score_raw']] = 90
        ideal[feat_idx['added_sugar']] = 0
        ideal[feat_idx['added_salt']] = 0
        ideal[feat_idx['preservatives']] = 0
        ideal[feat_idx['artificial_flavours']] = 0
        ideal[feat_idx['artificial_colors']] = 0
        ideal[feat_idx['trans_fat']] = 0
        ideal[feat_idx['protein']] = 15             # high protein ideal
        ideal[feat_idx['sugars']] = 5               # low sugar ideal
        ideal[feat_idx['fat']] = 8                  # low fat ideal
    elif health_pref == 'moderate':
        ideal[feat_idx['health_score_raw']] = 130   # mid-high
        ideal[feat_idx['eco_score_raw']] = 90
        ideal[feat_idx['protein']] = 7
        ideal[feat_idx['sugars']] = 20
        ideal[feat_idx['fat']] = 15
    else:  # any
        ideal[feat_idx['health_score_raw']] = 150
        ideal[feat_idx['eco_score_raw']] = 90

    # Apply slider values to ideal
    ideal[feat_idx['sugars']] = preferences.get('max_sugar', 50) * 0.5
    ideal[feat_idx['fat']] = preferences.get('max_fat', 35) * 0.5
    ideal[feat_idx['protein']] = preferences.get('min_protein', 0)

    # Scale ideal and compute cosine similarity only for filtered products
    ideal_scaled = SCALER.transform([ideal])[0] * 0.6
    ideal_full = np.concatenate([ideal_scaled, np.zeros(X_COMBINED.shape[1] - len(ideal_scaled))])

    for idx, pd in filtered:
        sim = float(cosine_similarity([ideal_full], [X_COMBINED[idx]])[0][0])
        pd['similarity'] = round(sim * 100, 1)

    # ── Step 6: Sort ──────────────────────────────────────────────────────────
    # Primary: health score descending (higher = better = A grade first)
    # Secondary: cosine similarity descending
    filtered.sort(key=lambda x: (-x[1]['health_score_raw'], -x[1]['similarity']))

    return [pd for _, pd in filtered[:top_n]]


def get_cluster_summary() -> list:
    """Return summary stats for each cluster."""
    summary = {}
    for i, p in enumerate(PRODUCTS):
        c = int(CLUSTERS[i])
        label = CLUSTER_LABELS[c]
        if label not in summary:
            summary[label] = {
                'label': label,
                'count': 0,
                'avg_health': [],
                'avg_protein': [],
                'bg': CLUSTER_COLORS[label],
                'color': CLUSTER_TEXT_COLORS[label],
            }
        summary[label]['count'] += 1
        summary[label]['avg_health'].append(p['health_score_raw'])
        summary[label]['avg_protein'].append(p['protein'] or 0)

    result = []
    for label, s in summary.items():
        result.append({
            'label': s['label'],
            'count': s['count'],
            'avg_health': round(np.mean(s['avg_health']), 1),
            'avg_protein': round(np.mean(s['avg_protein']), 1),
            'bg': s['bg'],
            'color': s['color'],
        })
    result.sort(key=lambda x: x['avg_health'])
    return result
