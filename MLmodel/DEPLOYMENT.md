# Backend Deployment Guide

## Deploy to Render

### Step 1: Push to GitHub
```bash
git add MLmodel/
git commit -m "Add Flask backend API"
git push
```

### Step 2: Deploy on Render
1. Go to https://render.com
2. Sign up / Login
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: ecom-sentiment-backend
   - **Root Directory**: `MLmodel`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free

6. Click **"Create Web Service"**

### Step 3: Wait for Deployment
- First deploy takes 5-10 minutes
- You'll get a URL like: `https://ecom-sentiment-backend.onrender.com`

### Step 4: Test Backend
```bash
curl https://your-backend-url.onrender.com/health
```

Should return: `{"status":"healthy"}`

### Step 5: Update Frontend
Update `frontend/src/app/api/analyze/route.ts`:

```typescript
const BACKEND_URL = 'https://your-backend-url.onrender.com';

// Call backend
const response = await fetch(`${BACKEND_URL}/analyze`, {
  method: 'POST',
  body: formData
});
```

---

## API Endpoints

### GET /
Health check and API info

### GET /health
Simple health check

### POST /analyze
Upload CSV and run full ML pipeline

**Request:**
- Content-Type: multipart/form-data
- Body: file (CSV file)

**Response:**
```json
{
  "status": "success",
  "results": {
    "total_reviews": 170,
    "sentiment_distribution": {"Positive": 120, "Negative": 30, "Neutral": 20},
    "positive_keywords": [...],
    "negative_keywords": [...],
    "aspect_sentiment": [...],
    "failure_components": [...],
    "top_products": [...]
  }
}
```

---

## Local Testing

```bash
cd MLmodel
python app.py
```

Visit: http://localhost:5000

Test upload:
```bash
curl -X POST http://localhost:5000/analyze -F "file=@data/amazon.csv"
```
