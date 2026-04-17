# Automatic Radiology Report Generator

A web application that takes a chest X-ray image as input and generates a written clinical report, similar to what a radiologist would produce. Built using an encoder-decoder architecture where a CNN extracts visual features from the scan and a language model translates those features into a structured medical report.

Dataset: [Indiana University Chest X-rays](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) (6,457 labeled samples after cleaning)


## How It Works

The system follows a two-stage pipeline:

1. **Image Encoding** -- A pre-trained convolutional neural network (ResNet50 / DenseNet121) processes the X-ray and outputs a fixed-length feature vector that captures the visual content of the scan.

2. **Report Decoding** -- That feature vector is passed into a language model (T5 / BLIP) which generates the report text word by word, producing both a "Findings" section and an "Impression" summary.

The web interface lets you upload an X-ray, sends it to a Flask backend for processing, and displays the generated report in the browser.


## Project Structure

```
Automatic-Report-Generator/
│
├── model/
│   ├── preproccessing.ipynb       # Data cleaning and preparation
│   ├── Real-X-Ray.ipynb           # Model training and evaluation (ResNet50 + T5)
│   └── processed_xray_data.csv    # Cleaned dataset (6,457 samples)
│
├── Model 2/
│   ├── app.py                     # Original Flask API (DenseNet121 + LSTM, needs .keras weights)
│   ├── server.py                  # Production Flask API (BLIP model from Hugging Face)
│   ├── preproccessing.ipynb       # Copy of preprocessing notebook
│   └── processed_xray_data.csv    # Copy of cleaned dataset
│
├── webpage/
│   ├── index.html                 # Main dashboard page
│   ├── style.css                  # Styling (dark theme, glassmorphism)
│   └── script.js                  # Frontend logic (upload, API calls, display)
│
├── arrg/                          # Python virtual environment (not committed)
├── preproccessing.ipynb           # Root copy of preprocessing notebook
├── processed_xray_data.csv        # Root copy of cleaned dataset
└── README.md
```


## File Breakdown

### What each file does

**model/preproccessing.ipynb** -- Reads the raw Kaggle CSV files (`indiana_reports.csv` and `indiana_projections.csv`), merges them on patient ID, keeps only the columns we need (filename, projection, findings, impression), drops rows with missing data, and saves the result as `processed_xray_data.csv`.

**model/Real-X-Ray.ipynb** -- The main training notebook. Defines an `XRayReportGenerator` class that combines a ResNet50 image encoder with a T5 text decoder. Trains for 5 epochs on the processed dataset, saves weights to `xray_report_model.pth`, and includes a `generate_report()` function for inference. Also contains BLEU score evaluation (BLEU-1 through BLEU-4).

**processed_xray_data.csv** -- The cleaned dataset with 6,457 rows. Each row has four columns: `filename` (the X-ray image file), `projection` (Frontal or Lateral), `findings` (the detailed observations), and `impression` (the summary diagnosis).

**Model 2/app.py** -- An alternative Flask backend that uses DenseNet121 for feature extraction and an LSTM-based decoder for caption generation. Requires `model_2.keras`, `wordtoix.pkl`, and `ixtoword.pkl` to run. Uses temperature scaling and top-k sampling during text generation.

**Model 2/server.py** -- The production Flask server that powers the web app. Loads a pre-trained BLIP model from Hugging Face (`Salesforce/blip-image-captioning-base`), serves the frontend files from the `webpage/` directory, and exposes a `/generate-report` POST endpoint that accepts an image and returns JSON with findings, impression, and confidence.

**webpage/index.html** -- The main page of the web interface. Has a drag-and-drop upload area, an image preview with a scanning animation, and a results panel that shows the generated report.

**webpage/style.css** -- All the visual styling. Dark theme, gradient text, glassmorphism cards, a pulsing status indicator, and a scanning-line animation that plays while the model processes the image.

**webpage/script.js** -- Handles file uploads via drag-and-drop or file picker, sends the image to the Flask API, displays the results, and falls back to sample report data if the backend is not running.


### What runs in production

The live application uses three files:

| File | Role |
|------|------|
| `Model 2/server.py` | Backend. Loads the AI model, handles image processing, serves the frontend. |
| `webpage/index.html` | Frontend structure. |
| `webpage/style.css` | Frontend styling. |
| `webpage/script.js` | Frontend logic and API communication. |

Everything else (notebooks, CSVs, `app.py`) was used during development and training.


## Getting Started

**Prerequisites:** Python 3.10+

```bash
# Clone the repo
git clone https://github.com/talwararyan07/Automatic-Report-Generator.git
cd Automatic-Report-Generator

# Create a virtual environment and install dependencies
python3 -m venv arrg
source arrg/bin/activate
pip install torch torchvision transformers flask flask-cors pillow pandas tqdm nltk

# Start the server
python "Model 2/server.py"
```

On the first run, the server will download the BLIP model weights (~990 MB). After that, it starts up in a few seconds.

Once running, open **http://localhost:5001** in your browser, upload a chest X-ray, and click "Generate Report."


## Architecture

```
                    +------------------+
  X-ray Image  -->  |  Image Encoder   |  -->  Feature Vector (visual summary)
   (224x224)        |  (ResNet / BLIP) |         |
                    +------------------+         |
                                                 v
                    +------------------+
                    | Language Decoder  |  -->  "The lungs are clear. No acute disease."
                    |   (T5 / BLIP)    |
                    +------------------+
```

The encoder converts the image into a numerical representation. The decoder reads that representation and generates text one word at a time until it forms a complete report.


## Evaluation

The model was evaluated using BLEU scores on the test set:

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.7096 |
| BLEU-2 | 0.5390 |
| BLEU-3 | 0.4296 |
| BLEU-4 | 0.3225 |


## Tech Stack

**Backend:** Python, Flask, PyTorch, Hugging Face Transformers

**Frontend:** HTML, CSS, JavaScript (no frameworks)

**Models explored:** ResNet50, DenseNet121, T5, LSTM, BLIP

**Dataset:** Indiana University Chest X-ray Collection (Kaggle)


## Authors

Built as a collaborative project. Contributions welcome.
