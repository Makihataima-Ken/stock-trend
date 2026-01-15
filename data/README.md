# Dataset Setup

This project uses the Kaggle competition dataset:

**Predicting Stock Trends: Rise or Fall**  
https://www.kaggle.com/competitions/predicting-stock-trends-rise-or-fall

⚠️ **Important**
- The dataset is **not included** in this repository.
- This is intentional and required to comply with Kaggle’s competition rules.
- You must download the data yourself after enrolling in the competition.

---

## How to Set Up the Dataset

### 1. Create a Kaggle Account
Go to https://www.kaggle.com and create an account if you don’t already have one.

### 2. Enroll in the Competition
Visit the competition page and click **“Join Competition”**.

### 3. Download the Dataset
After enrolling:
- Go to the **Data** tab of the competition
- Download the dataset files

### 4. Extract Files into This Folder

After extraction, this directory should look like:
```
data/
├── train.csv
├── test.csv
└── README.md
```

Do **not** rename the files unless you also update the notebook code.

---

## Notes
- The `data/` directory is ignored by Git and should never be committed.
- Each team member must download their own copy of the dataset.
- The notebook will fail with a clear error if the dataset is missing.

If you are running the project on **Kaggle**, this folder is not needed because Kaggle mounts the dataset automatically.