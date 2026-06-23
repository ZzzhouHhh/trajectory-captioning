# Synthesizing Geographical Semantics and Large Language Models for OD-based Trajectory Captioning

Code and data for our trajectory captioning method. This work focuses on trajectory captioning from origin-destination (OD) pairs.

## Directory Structure

```
resoures/
├── README.md
├── requirements.txt
│
├── utils/                           # shared utilities
│   ├── spatial.py                   #   Haversine distance, grid indexing, GPS noise
│   └── temporal.py                  #   time encoding (sin/cos), weekday mapping
│
├── data_preprocessing/              # data loading & preprocessing
│   ├── sample_clean.py              #   trajectory sampling, column cleaning
│   ├── poi_match.py                 #   POI nearest-neighbor matching, info joining
│   └── build_tky_traj.py            #   TKY trajectory construction from check-ins
│
├── embedding/                       # time & subcategory embedding
│   ├── time_embed.py                #   time node embedding (one-hot + sin/cos)
│   └── category_embed.py            #   MiniLM category text encoding
│
├── context_augmentation/            # CAM: Context Augmentation Module
│   ├── popularity.py                #   time period popularity P_k
│   ├── uniqueness.py                #   category uniqueness U_k
│   ├── co_calculation.py            #   co_k = (w*P_k+(1-w)*U_k)/(1+dist)
│   └── augment.py                   #   full augmentation pipeline
│
├── time_subcategory_gnn/            # GSRM: Time-Subcategory GCN
│   ├── build_graph.py               #   heterogeneous graph construction
│   ├── gcn_model.py                 #   GCN layers + TimeCategoryGCN model
│   └── train.py                     #   training loop + loss functions
│
├── poi_matching/                    # GSRM: POI Matching
│   ├── models.py                    #   POIModel, POIClassifier, etc.
│   ├── dataset.py                   #   POIRecommendDataset
│   ├── train.py                     #   training + evaluation
│   └── inference.py                 #   predict_poi_id, batch matching
│
├── text_generation/                 # TDM: Trajectory Decoding Module
│   ├── prompt.py                    #   CoT prompt templates, formatting
│   ├── decoder.py                   #   LLM loading, generate_caption
│   └── batch_generate.py            #   batch caption generation
│
├── evaluation/                      # evaluation metrics
│   └── metrics.py                   #   BLEU, ROUGE, BERTScore computation
│
├── ablation/                        # ablation study
│   ├── module_ablation.py           #   w/o c&g, w/o c, w/o g, w/o pp, w/o CoT
│   └── prompt_ablation.py           #   prompt component ablation
│
└── analysis/                        # hyperparameter analysis
    ├── sweep_w.py                   #   ω parameter sweep
    ├── sweep_d.py                   #   d (GPS offset) parameter sweep
    └── draw_pic.py                  #   visualization (ω/d effect, cluster plots)
```

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+, PyTorch 2.0+, CUDA GPU recommended.

## Running the Pipeline

Run each step in order. Output from each step feeds into the next.

### Prerequisite: Data Preparation

Place the following CSV files under `data/`:

| File | Required Columns |
|------|-----------------|
| `nyc_taxi_trips.csv` | `tpep_pickup_datetime, tpep_dropoff_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count` |
| `nyc_checkins.csv` | `userId, poi_id, lat, lon, utcTimestamp, weekday, hour, venueCategory, classification` |
| `poi_info_batch.csv` | `poi_id, name, category, venueCategory, latitude, longitude` |

### Step 1: Trajectory Sampling & Cleaning

```bash
python -c "
from data_preprocessing import random_sample_from_csv, clean_and_select_columns
df = random_sample_from_csv('data/nyc_taxi_trips.csv', sample_size=100000)
cols = ['passenger_count','tpep_pickup_datetime','tpep_dropoff_datetime',
        'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','total_amount']
cleaned = clean_and_select_columns(df, cols, output_path='output/cleaned_trajectories.csv')
"
```

### Step 2: Context Augmentation (CAM)

```bash
python -c "
import pandas as pd
from context_augmentation import augment_trajectory_context
traj = pd.read_csv('output/cleaned_trajectories.csv')
checkin = pd.read_csv('data/nyc_checkins.csv')
augmented = augment_trajectory_context(traj, checkin, num_categories=9, w=0.4)
augmented.to_csv('output/augmented_trajectories.csv', index=False)
"
```

### Step 3: Time-Subcategory GCN Training (GSRM)

```bash
python -c "
import pandas as pd, torch
from time_subcategory_gnn import build_graph_from_checkins, train_time_subcategory_gcn
df = pd.read_csv('data/nyc_checkins.csv')
G = build_graph_from_checkins(df, k=5)
model, node_to_id, id_to_node, adj_tensor, node_features = \
    train_time_subcategory_gcn(df, G, num_epochs=200, device='cuda:0')
torch.save(model.state_dict(), 'output/gcn_model.pth')
"
```

### Step 4: POI Matching Training (GSRM)

```bash
python -c "
from poi_matching import main_train_pipeline
model1, poi_model, scaler, label_encoder = main_train_pipeline(
    'data/nyc_checkins.csv', device='cuda:0'
)
"
```

### Step 5: Batch Caption Generation (TDM)

```bash
python -c "
import pandas as pd
from text_generation import load_llm, batch_generate_captions
model, tokenizer = load_llm('meta-llama/Meta-Llama-3.1-8B-Instruct', device='cuda:0')
traj_df = pd.read_csv('output/augmented_trajectories.csv')
result = batch_generate_captions(model, tokenizer, traj_df, use_cot=True)
result.to_csv('output/generated_captions.csv', index=False)
"
```

### Step 6: Evaluation

```bash
python -c "
import pandas as pd
from evaluation import evaluate_all
gen = pd.read_csv('output/generated_captions.csv')['generated_caption'].tolist()
ref = pd.read_csv('data/reference_captions.csv')['caption'].tolist()
evaluate_all(gen, ref)
"
```

### Step 7: Module Ablation Study

```bash
python -c "
import pandas as pd
from ablation import run_full_ablation_study
traj = pd.read_csv('output/augmented_trajectories.csv')
checkin = pd.read_csv('data/nyc_checkins.csv')
references = pd.read_csv('data/reference_captions.csv')['caption'].tolist()
run_full_ablation_study(traj, checkin, references)
"
```



## Datasets

| Dataset | Trajectories | Check-ins | POIs | Subcategories |
|---------|-------------|-----------|------|---------------|
| NYC | >13M taxi trips | 227,428 | 38,334 | 251 |
| TKY | 4,522 simulated | 573,703 | 61,858 | 247 |

## Data

The annotated reference captions are included in the `data/` directory.

We note that the reference captions represent plausible semantic interpretations inferred from observable trajectory attributes rather than objective ground-truth descriptions of passenger intent.




