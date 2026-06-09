# NYC Taxi Fare Prediction — Multi-Source ML Pipeline
**Regression & Deep Learning | Python · Scikit-learn · TensorFlow · Pandas**

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)](https://tensorflow.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Regression-green?style=flat-square)](https://scikit-learn.org)
[![Data](https://img.shields.io/badge/Data-NYC%20TLC%20Jan%202019-yellow?style=flat-square)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

---

## 1. Business Problem

NYC taxi operators and ride-hailing platforms need accurate fare estimates before a trip begins — without knowing how long it will take or which route the driver will follow. A pre-trip fare model could power:

- **Upfront pricing** for passengers (like Uber's fixed price feature)
- **Demand forecasting** by zone, time, and weather condition
- **Driver dispatch optimisation** — routing drivers toward high-fare zones at the right time

> *"Can the total fare of an NYC taxi trip be predicted reliably using only the information available at the moment of pickup — location, time of day, weather, and calendar context — without knowing the trip distance in advance?"*

This project builds a full ML pipeline from raw TLC trip records through to a tuned ensemble model and a TensorFlow neural network, comparing performance across five model types.

---

## 2. Data Sources

Three datasets were joined to construct the final feature matrix:

| Source | Description | Key Fields |
|--------|-------------|-----------|
| NYC TLC Yellow Taxi (Jan 2019) | Official trip records — 7.6M+ rows | Pickup datetime, dropoff datetime, pickup/dropoff location IDs, passenger count, trip distance, payment type, total amount |
| NYC Taxi Zone Lookup | Maps 265 location IDs to borough and zone name | LocationID, Borough, Zone |
| NYC Weather 2019 (daily) | Historical weather for New York City | Temperature (°F), precipitation (in), cloud cover (%), humidity (%), wind speed (mph) |

**Data joining logic:**
- Trip records → Zone lookup joined on `PULocationID = LocationID` → adds `Borough` per trip
- Trip records → Weather joined on `transaction_date = weather_date` → adds daily weather context

---

## 3. Data Cleaning

The raw dataset required significant cleaning before modelling:

```
Raw: 7,656,792 rows
  ↓  Remove total_amount < 0 (refunds/errors, linked to dispute payment types)
  ↓  Remove total_amount > 200 (extreme outlier removal — 0.02% of data)
  ↓  Drop nulls (passenger_count, RatecodeID had sparse nulls)
Cleaned: ~7.6M rows retained
```

**Key data quality findings:**
- Negative fares were almost exclusively `payment_type = 3` (no charge) or `payment_type = 4` (dispute) — not data errors, but legitimate zero-revenue trips excluded from fare prediction scope
- Zero-fare trips clustered around 0-mile distances — likely cancelled or test records
- Outliers above $200 were genuine airport/long-distance fares excluded to prevent model distortion on the core urban fare distribution

---

## 4. Feature Engineering

The base dataset was enriched with five categories of engineered features:

### 4.1 Datetime Features
```python
trip_data_prepared['transaction_date']  = pickup_datetime.dt.date
trip_data_prepared['transaction_year']  = pickup_datetime.dt.year
trip_data_prepared['transaction_month'] = pickup_datetime.dt.month
trip_data_prepared['transaction_day']   = pickup_datetime.dt.day
trip_data_prepared['transaction_hour']  = pickup_datetime.dt.hour
trip_data_prepared['transaction_week_day'] = transaction_date.dt.weekday
trip_data_prepared['weekend'] = weekday.apply(lambda x: True if x in [5, 6] else False)
```

### 4.2 Holiday Flag
```python
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2018', end='2020').date
data['is_holiday'] = data['transaction_date'].isin(holidays)
```

### 4.3 Borough Mapping
Joined the TLC zone lookup table to map each `PULocationID` to one of NYC's five boroughs (Manhattan, Brooklyn, Queens, Bronx, Staten Island + EWR/Unknown). This collapses 265 location IDs into a manageable categorical with strong geographic signal.

### 4.4 Weather Features
Five daily weather metrics joined by date:
- `temperature_F` — ambient temperature
- `precipitation_in` — rainfall/snowfall
- `cloud_cover_%` — overcast conditions
- `humidity_%` — humidity level
- `windspeed_mph` — wind conditions

### 4.5 Aggregation
Trips were grouped by `[PULocationID, date, month, day, hour]` and mean-aggregated to produce one row per location-time combination, with `count_of_transaction` added as a volume signal. This reduces the 7.6M trip rows to a manageable analytical dataset while preserving spatial and temporal granularity.

---

## 5. Modelling Approach

```
Cleaned, feature-engineered dataset
        ↓
Train/Test split (67% / 33%, random_state=42)
        ↓
One-hot encode categorical features (PULocationID, Borough, weekday, etc.)
        ↓  280 input features after encoding
┌─────────────────────────────────────────────────┐
│  Benchmark: Decision Tree (no trip_distance)     │
│  Model 1:   Decision Tree (max_depth=10)         │
│  Model 2:   Random Forest (default)              │
│  Model 3:   Gradient Boosting (default)          │
│  Model 4:   Random Forest (RandomizedSearchCV)   │
│  Model 5:   Neural Network (TensorFlow/Keras)    │
└─────────────────────────────────────────────────┘
        ↓
Evaluation: MAE, RMSE, R²
        ↓
Model comparison & recommendation
```

### Neural Network Architecture
```
Input: 280 features (StandardScaler normalised)
  → Dense(256, ReLU) → Dropout(0.2)
  → Dense(128, ReLU) → Dropout(0.2)
  → Dense(64, ReLU)
  → Dense(1)  [regression output]

Loss: Huber (robust to outliers vs MSE)
Optimiser: Adam
Early stopping: patience=10, restore best weights
Batch size: 256 | Max epochs: 100
```

### Hyperparameter Tuning (Random Forest)
RandomizedSearchCV with 3-fold CV, 10 iterations over:
- `n_estimators`: 200–2000
- `max_depth`: 10, 20, 50, 100, 150, 200, 300, 500
- `max_features`: sqrt, auto
- `min_samples_split`: 2, 5, 10, 20, 40
- `min_samples_leaf`: 1, 2, 4, 10, 20
- `bootstrap`: True / False

---

## 6. Model Performance

| Model | MAE ($) | RMSE ($) | R² |
|-------|---------|---------|-----|
| Decision Tree (baseline) | ~9.2 | ~14.8 | ~0.28 |
| Random Forest (default) | ~7.4 | ~12.7 | ~0.40 |
| Gradient Boosting | ~7.3 | ~12.5 | ~0.41 |
| **Random Forest #1 (tuned)** | **7.07** | **12.42** | **0.417** |
| Random Forest #2 (tuned alt.) | 7.15 | 12.36 | 0.422 |
| Neural Network (TensorFlow) | 7.20 | 12.87 | 0.374 |

**Winner: Tuned Random Forest** — best MAE at $7.07, and R² of 0.417–0.422.

---

## 7. Key Findings

### Finding 1 — Tree ensembles beat the neural network on this task
The tuned Random Forest outperformed the 4-layer TensorFlow neural network on all three metrics. This is consistent with literature on tabular data: tree ensembles typically outperform deep learning on structured, mixed-type datasets without large sample sizes per feature combination.

### Finding 2 — Trip hour and borough are the top predictors
From Gradient Boosting feature importance analysis, `transaction_hour` and `Borough` (especially Manhattan vs outer boroughs) are the two strongest predictors of total fare. This is consistent with NYC fare dynamics: Manhattan peak-hour trips attract surcharges, congestion pricing, and longer distance accumulation.

### Finding 3 — Weather enrichment improved accuracy by ~18% over location-only baseline
Adding the five weather variables (temperature, precipitation, cloud cover, humidity, wind speed) improved R² from ~0.35 to ~0.42 compared to a location + time only model. Precipitation days show measurably higher average fares, likely due to demand spikes and slower traffic increasing metered time.

### Finding 4 — The holiday flag had minimal independent effect
`is_holiday` added little predictive power once `transaction_hour` and `Borough` were included — holidays affect *when* people travel, but the fare effect is captured by the existing time and location features.

### Finding 5 — R² ceiling reflects genuine fare unpredictability
An R² of ~0.42 is expected for taxi fares without knowing trip distance. The remaining variance is driven by real-time traffic (unpredictable at booking time), route choice, and MTA toll variation. This is not model failure — it is the irreducible noise floor when predicting fare from pre-trip context only.

---

## 8. Recommendations

| For | Recommendation |
|-----|---------------|
| Ride-hailing product teams | Deploy tuned Random Forest as the upfront fare estimator; retrain monthly as seasonal patterns shift |
| Pricing strategy | Apply dynamic fare multiplier on precipitation days — data supports a measurable surge |
| Driver dispatch | Prioritise Manhattan zone pickups during hours 7–9 and 17–20 for highest expected fare per trip |
| Data engineering | Replace daily weather aggregation with hourly weather joins — would likely close remaining R² gap |
| Future work | Add traffic congestion index (e.g., Google Maps API real-time data) as a feature — the single largest missing predictor |

---

## 9. Files in This Repository

```
├── New york taxi project.ipynb     # Full analysis notebook
└── README.md
```

**External data (not in repo — download links):**
- [NYC TLC Yellow Taxi Jan 2019](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — `yellow_tripdata_2019-01.parquet`
- [Taxi Zone Lookup](https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv) — `taxi_zone_lookup.csv`
- NYC Weather 2019 — available via NOAA or Visual Crossing Weather API

---

## 10. Tools & Libraries

| Tool | Use |
|------|-----|
| Pandas / NumPy | Data wrangling, feature engineering, datetime extraction |
| Matplotlib | EDA visualisations, model comparison bar charts |
| Scikit-learn | Decision Tree, Random Forest, Gradient Boosting, RandomizedSearchCV, StandardScaler, metrics |
| TensorFlow / Keras | Neural network (Sequential API, Dense, Dropout, EarlyStopping) |
| `pandas.tseries.holiday` | US Federal Holiday calendar for `is_holiday` flag |

---

## 11. How to Run

```bash
# Clone the repository
git clone https://github.com/jamesenet/New-york-taxi-project

# Install dependencies
pip install pandas numpy matplotlib scikit-learn tensorflow pyarrow

# Download source data (see Section 9 for links)
# Update local file paths in cell 3 of the notebook

# Launch notebook
jupyter notebook "New york taxi project.ipynb"
```

> **Note:** The Random Forest tuning step (RandomizedSearchCV, n_iter=10, cv=3) is computationally intensive.
> On a standard laptop expect 15–30 minutes. Set `n_jobs=-1` to use all available CPU cores.

---

*Case study by Cornelius Enetomhe · [LinkedIn](https://www.linkedin.com/in/cornelius-enetomhe-01688a266/) · [Portfolio](https://jamesenet.github.io/CorneliusEnetomhe.github.io/)*
