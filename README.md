# DSA 4060: Recommender Systems - Week 3 Lab
## Content-Based Movie Recommendation Engine

- **Course:** DSA 4060: Recommender Systems
- **Assignment:** Lab Assignment 3 – Content-Based Filtering
- **Student Name:** Nicholas Kinyanjui
- **Student ID:** 670178
- **Repository Name:** `DSA4060-Week3-670178`

---

## 1. Project Overview & Objective

This repository contains the complete implementation, analysis, and evaluation of an end-to-end **Content-Based Recommendation Engine** developed in Python using `pandas` and `numpy`.

The objective of content-based filtering is to recommend items to a user based on the descriptive attributes (item profile) of items that the user has rated highly in the past. This project demonstrates how structured metadata is transformed into vector representations, weighted by explicit user ratings, normalized to formulate a personalized user preference profile, and used to rank unseen items using dot-product similarity scoring.

---

## 2. Repository Structure

```text
DSA4060-Week3-670178/
├── content_based_lab.ipynb   # Fully documented and executed Jupyter Notebook
├── movies.csv                # Dataset of 10 movies across 5 binary genre features
├── README.md                 # Complete documentation, theory, results, and answers
└── generate_notebook.py      # Automated generation and execution pipeline script
```

---

## 3. Content-Based Filtering Workflow

The engine follows an 8-stage algorithmic pipeline:

```text
┌─────────────────┐     ┌────────────────┐     ┌──────────────┐     ┌──────────────┐
│  Item Metadata  │ ──> │ Item Profiles  │ ──> │ User Ratings │ ──> │ User Profile │
│  (movies.csv)   │     │ (Binary Vector)│     │(1-5 Star Exp)│     │ (Raw Sum W)  │
└─────────────────┘     └────────────────┘     └──────────────┘     └──────────────┘
                                                                           │
                                                                           ▼
┌─────────────────────────┐     ┌──────────────┐     ┌──────────────┐┌──────────────┐
│ Top-N Recommendations   │ <── │ Remove Seen  │ <── │ Rank Scores  ││  Normalize   │
│  (Filtered & Ranked)    │     │(Filter Watch)│     │ (Desc Order) ││ (Unit Sum)   │
└─────────────────────────┘     └──────────────┘     └──────────────┘└──────────────┘
```

1. **Item Metadata (`movies.csv`):** Tabular catalog containing movie identifiers, titles, and genre flags.
2. **Item Profiles:** $k$-dimensional feature vectors $\mathbf{v}_i \in \{0, 1\}^k$ representing genre presence.
3. **User Ratings:** Explicit user ratings $r_{ui} \in [1, 5]$.
4. **Weighted Features:** Feature vectors scaled by user rating feedback: $\mathbf{W}_i = r_{ui} \cdot \mathbf{v}_i$.
5. **User Profile:** Aggregate preference vector $\mathbf{u}_{raw} = \sum_{i} \mathbf{W}_i$.
6. **Normalization:** Proportional scaling $\mathbf{u}_{norm} = \frac{\mathbf{u}_{raw}}{\sum_j u_{raw, j}}$ summing to 1.0.
7. **Similarity Scoring & Ranking:** Dot product score for every movie $\text{Score}_i = \mathbf{v}_i \cdot \mathbf{u}_{norm}$, sorted in descending order.
8. **Filtering & Top-N:** Exclusion of already-watched movies to produce the final recommendation list.

---

## 4. Key Experiments & Lab Findings

### 4.1 User A (Baseline Experiment)
- **Watched Movies:**
  - Movie 1: *Galaxy Mission* (Rating: 5, Action=1, SciFi=1)
  - Movie 3: *Cosmic Odyssey* (Rating: 4, Drama=1, SciFi=1)
  - Movie 5: *Final Heist* (Rating: 4, Action=1, Drama=1)
  - Movie 7: *Cyberpunk 2099* (Rating: 5, Action=1, SciFi=1)
- **Learned Preference Profile:**
  - Action: $14$ ($38.89\%$)
  - SciFi: $14$ ($38.89\%$)
  - Drama: $8$ ($22.22\%$)
  - Comedy: $0$ ($0.0\%$)
  - Romance: $0$ ($0.0\%$)
- **Top 3 Recommendations for User A:**
  1. **Alien Encounter** (Score: `0.7778`) — Matches both Action and SciFi.
  2. **Romantic Sunset** (Score: `0.2222`) — Contains Drama.
  3. **Heartfelt Melodies** (Score: `0.2222`) — Contains Drama.

---

### 4.2 User B (Divergent Tastes Comparison)
- **Ratings:** Movies 2, 4, 6, and 9 (Rated 5, 4, 5, 4) — Heavy affinity for Romance and Comedy.
- **Top 3 Recommendations for User B:**
  1. **Heartfelt Melodies** (Score: `0.5938`) — Drama + Romance.
  2. **Cosmic Odyssey** (Score: `0.1562`) — Drama + SciFi.
  3. **Final Heist** (Score: `0.1562`) — Action + Drama.
- **Comparison Conclusion:** User A and User B received distinct, non-overlapping recommendations reflecting their respective preference distributions.

---

### 4.3 Thinking Exercise: User C (Single Rating Edge Case)
- **Scenario:** User C watched only *Galaxy Mission* and gave it 5 stars.
- **Learned Profile:** 50% Action, 50% SciFi, 0% all other genres.
- **System Failure Risks:**
  - *Extreme Overspecialization:* The user is trapped in an echo chamber of Action and SciFi.
  - *Cold-Start Distortion:* The algorithm assumes a single rating represents the user's complete worldview.

---

### 4.4 Custom Additions (Nicholas Kinyanjui)
- **Custom Movie Added:**
  - `movie_id`: 11
  - `title`: *"Nairobi Nights: Urban Pulse"*
  - Features: Comedy = 1, Drama = 1, Action = 0, Romance = 0, SciFi = 0
- **Nicholas's Ratings:**
  - Movie 1 (*Galaxy Mission*): 5 stars
  - Movie 3 (*Cosmic Odyssey*): 4 stars
  - Movie 6 (*Romantic Sunset*): 2 stars
  - Movie 9 (*Funny Neighbors*): 3 stars
- **Nicholas's Preference Profile:**
  - SciFi: `0.3214` (32.14%)
  - Drama: `0.2143` (21.43%)
  - Action: `0.1786` (17.86%)
  - Romance: `0.1786` (17.86%)
  - Comedy: `0.1071` (10.71%)
- **Top 5 Recommendations for Nicholas:**
  1. **Cyberpunk 2099** (Score: `0.5000`)
  2. **Alien Encounter** (Score: `0.5000`)
  3. **Final Heist** (Score: `0.3929`)
  4. **Heartfelt Melodies** (Score: `0.3929`)
  5. **Nairobi Nights: Urban Pulse** (Score: `0.3214`)

---

## 5. Lab Questions & Answers

### Question 1: What is the shape of the dataset and what do the binary values (1 and 0) represent?
- **Answer:** The dataset shape is $10 \times 7$ (10 movies, 7 attributes). The values `1` and `0` represent binary genre membership (multi-hot encoding): `1` denotes that the movie belongs to that genre, while `0` denotes its absence.

### Question 2: Why are these 5 features considered "structured"?
- **Answer:** They conform to a fixed, tabular schema with discrete values and clear semantic definitions. Unlike unstructured text or media, structured features can be directly operated upon using linear algebra (dot products and vector addition).

### Question 3: How does the item profile represent an item in vector space?
- **Answer:** Each item is a vector $\mathbf{v}_i \in \mathbb{R}^k$ in a $k$-dimensional Euclidean space where coordinate axes represent individual genres. Vector distance or angle measures semantic similarity between items.

### Question: Which two genres appear strongest for User A? Explain using profile scores.
- **Answer:** **Action** (14) and **SciFi** (14). User A rated *Galaxy Mission* (5), *Cosmic Odyssey* (4), *Final Heist* (4), and *Cyberpunk 2099* (5). Action and SciFi each accumulated 14 weighted points across these titles.

### Questions 4–7 (Thinking Exercise - User C):
- **4. What will the model learn about User C?** It infers that User C only cares about Action and SciFi, assigning 0 weight to all other genres.
- **5. Which genres will dominate the profile?** Action (50%) and SciFi (50%).
- **6. What could go wrong with recommendations?** The model suffers from extreme overspecialization (filter bubble) and cannot recommend any diverse or serendipitous content.
- **7. Is one rating enough to understand preferences?** No. One observation suffers from high statistical variance and fails to capture multi-faceted human tastes.

### Questions 8–12 (Interpretation for Nicholas Kinyanjui):
- **8. What are your user's two strongest preferences?** **SciFi** (0.3214) and **Drama** (0.2143).
- **9. Which three movies received the highest recommendation scores?** *Cyberpunk 2099* (0.5000), *Alien Encounter* (0.5000), and *Final Heist* (0.3929).
- **10. Why did the top movie rank first?** *Cyberpunk 2099* contains Action and SciFi, which are Nicholas's #1 and tied #3 preferences, combining to produce the top similarity score of 0.5000.
- **11. Did any recommendation surprise you?** Yes, *Heartfelt Melodies* (Drama + Romance) scored 0.3929 despite Nicholas awarding only 2 stars to *Romantic Sunset*. This occurred because low ratings still add positive weights rather than penalizing features.
- **12. What important information is the model ignoring?** It ignores negative rating penalties, directors, actors, release year, user reviews, watch time, and temporal recency.

---

## 6. Limitations of the Model

1. **Coarse Feature Granularity:** Binary genre flags ignore pacing, director style, cinematography, and plot nuance.
2. **Filter Bubbles & Serendipity Deficit:** The model never suggests novel genres outside the user's explicit history.
3. **Cold-Start Problem:** New users with zero ratings receive no recommendations ($\mathbf{u} = \mathbf{0}$).
4. **No Negative Preference Penalization:** Low ratings (1 or 2 stars) still add positive preference points instead of subtracting.
5. **Metadata Dependence:** Missing or inaccurate metadata directly corrupts the quality of recommendations.

---

## 7. How to Run

### Option A: Using `uv` (Fastest, zero-config)
```bash
# Clone the repository
git clone https://github.com/your-username/DSA4060-Week3-670178.git
cd DSA4060-Week3-670178

# Run Jupyter Lab with all dependencies automatically installed
uv run --with jupyterlab --with pandas --with numpy jupyter lab
```

### Option B: Using standard Python / virtualenv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy jupyterlab
jupyter lab content_based_lab.ipynb
```
