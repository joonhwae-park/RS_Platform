# Docker Setup for P5 Model

This guide explains how to prepare the t5-small model files for the Docker image.

## Prerequisites

- Python 3.7+
- pip
- Docker

## Step 1: Install Required Python Packages

```bash
pip install transformers torch
```

## Step 2: Download t5-small Model Files

Run the provided script:

```bash
python download_t5_model.py
```

This will:
1. Download the t5-small model from HuggingFace
2. Extract the files to `./models/t5-small/`
3. Clean up temporary cache files

**Expected output:**
- A `models/t5-small/` directory containing:
  - `config.json`
  - `pytorch_model.bin` or `model.safetensors`
  - `tokenizer.json`
  - `spiece.model`
  - `tokenizer_config.json`
  - Other supporting files

## Step 3: Verify Model Files

Check that the files are in place:

```bash
ls -lh models/t5-small/
```

You should see several files totaling around 200-300 MB.

## Step 4: Build Docker Image

```bash
docker build -t recs-api .
```

The Dockerfile will:
1. Install all dependencies
2. Clone the P5_mod repository
3. Copy your `app.py`
4. **Copy t5-small model files into the HuggingFace cache location**
5. Set up the service to run on port 8080

## Step 5: Deploy to Google Cloud Run

```bash
# Tag the image
docker tag recs-api gcr.io/YOUR_PROJECT_ID/recs-api

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/recs-api

# Deploy to Cloud Run
gcloud run deploy recs-api \
  --image gcr.io/YOUR_PROJECT_ID/recs-api \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300
```

## Important Notes

1. **Model files are baked into the image**: The t5-small model is copied during Docker build, so no runtime download is needed.

2. **P5 checkpoint**: The `mvt_aug_epoch10.pth` file should already be available at `/models/p5/mvt_aug_epoch10.pth` in your Cloud Run environment.

3. **local_files_only flag**: The code now uses `local_files_only=True` when loading the model, preventing any HuggingFace API calls.

4. **No HuggingFace token needed**: Since the model is pre-loaded, you don't need to set `HF_TOKEN`.

## Troubleshooting

### "No such file or directory" error when building Docker
- Make sure `models/t5-small/` exists and contains model files
- Run `python download_t5_model.py` again

### "Rate limit" errors from HuggingFace
- This shouldn't happen anymore since the model is pre-loaded
- If it does, check that the COPY command in Dockerfile is correct

### Model files too large for build context
- The t5-small model is ~250MB, which is acceptable
- If you have issues, ensure `.dockerignore` excludes unnecessary files
