# Comprehensive Diagnosis: Why Recommendations Don't Work

## Root Cause Identified ✅

**The frontend is NOT calling the backend API because `VITE_RECOMMENDATION_API_URL` is missing from `.env`**

### Evidence

1. **Missing Environment Variable**
   - `.env` file only has Supabase credentials
   - Missing: `VITE_RECOMMENDATION_API_URL` (required)
   - Missing: `VITE_WEBHOOK_SECRET` (optional)

2. **Frontend Behavior When URL is Missing**
   - `recommendationAPI.ts` line 31-34: Returns `null` if URL not configured
   - `App.tsx` line 675-682: Falls back to `phase2_movies` when API call fails
   - Result: First 10 movies from database, slightly shuffled

3. **Backend is Ready**
   - Logs show: "Full initialization completed successfully" ✅
   - Service is healthy and waiting for requests
   - But NO requests are being received because frontend can't reach it

## How the System Should Work

```
User completes ratings
    ↓
Frontend calls: triggerRecommendationGeneration(sessionId)
    ↓
POST request to: ${VITE_RECOMMENDATION_API_URL}/recommend
    ↓
Backend generates recommendations (SVD + P5)
    ↓
Backend saves to recommendations table
    ↓
Frontend queries recommendations table by display_order
    ↓
User sees personalized recommendations
```

## What's Actually Happening

```
User completes ratings
    ↓
Frontend calls: triggerRecommendationGeneration(sessionId)
    ↓
VITE_RECOMMENDATION_API_URL is undefined
    ↓
API call returns null (line 34 of recommendationAPI.ts)
    ↓
Frontend uses fallback: getFallbackRecommendations()
    ↓
Queries phase2_movies table, gets first 10, shuffles them
    ↓
User sees generic movies (not personalized)
```

## Solution

### Step 1: Get Your Cloud Run Service URL

Based on your logs, your service is:
- **Service Name**: `recs-api`
- **Project**: `fourth-flag-470220-a9`
- **Region**: `us-central1`

The URL should be:
```
https://recs-api-<hash>-uc.a.run.app
```

To find the exact URL, run:
```bash
gcloud run services describe recs-api \
  --region=us-central1 \
  --project=fourth-flag-470220-a9 \
  --format='value(status.url)'
```

Or check in Google Cloud Console:
1. Go to Cloud Run
2. Click on "recs-api" service
3. Copy the URL at the top of the page

### Step 2: Update `.env` File

Add these lines to `/tmp/cc-agent/55569873/project/.env`:

```bash
# Recommendation API Configuration
VITE_RECOMMENDATION_API_URL=https://recs-api-YOUR-HASH-uc.a.run.app
VITE_WEBHOOK_SECRET=your_webhook_secret_here  # Optional, if you have one configured
```

**IMPORTANT**:
- Replace `YOUR-HASH` with the actual hash from your Cloud Run URL
- If you don't have a webhook secret configured in Cloud Run, you can either:
  - Leave this line out (the API will work without it)
  - Or set one in both places for security

### Step 3: Rebuild the Frontend

The environment variables are embedded at build time for Vite:

```bash
npm run build
```

### Step 4: Deploy the Updated Frontend

Deploy the new `dist/` folder to your hosting service.

### Step 5: Test the Connection

1. **Check API Health**
   - Open browser console
   - Go to your app
   - Look for: "Checking API health at: https://recs-api-..."
   - Should see: "Health check successful: {ok: true, status: 'ready_full', ...}"

2. **Complete a Rating Session**
   - Rate 5+ movies
   - Click "Get Recommendations"
   - In console, should see:
     ```
     ✅ Recommendation generation successful
     Recommendation API response: {session_id: "...", svd_top_saved: 10, p5_top_saved: 10, ...}
     ```

3. **Check Database**
   ```sql
   SELECT COUNT(*) FROM recommendations;
   ```
   Should be > 0 now!

## Verification Checklist

After deploying the fix:

- [ ] `.env` has `VITE_RECOMMENDATION_API_URL` configured
- [ ] Frontend rebuilt with `npm run build`
- [ ] Frontend deployed
- [ ] Browser console shows API URL is configured
- [ ] Health check succeeds
- [ ] Completing a rating session triggers API call
- [ ] Database `recommendations` table has data
- [ ] Users see personalized recommendations (not just shuffled phase2_movies)

## Quick Diagnostic Commands

**Check if URL is configured in built files:**
```bash
grep -r "recs-api" dist/assets/*.js
```

Should see your Cloud Run URL if configured correctly.

**Test API directly:**
```bash
# Replace with your actual Cloud Run URL
CLOUD_RUN_URL="https://recs-api-YOUR-HASH-uc.a.run.app"

# Health check
curl "${CLOUD_RUN_URL}/health"

# Should return:
# {"ok":true,"status":"ready_full","device":"cpu",...}
```

**Test recommendation generation (replace session_id with real one):**
```bash
curl -X POST "${CLOUD_RUN_URL}/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "a05f020e-c391-4ba9-bfb3-4047de64816e",
    "topk_per_model": 10,
    "phase": 2
  }'
```

## Why This Wasn't Caught Earlier

1. **Backend logs looked fine** - Service initialized successfully
2. **Frontend had no errors** - It gracefully falls back when API unavailable
3. **Users still got movies** - Just not personalized ones
4. **Environment variables are easy to miss** - Especially when they need to be set in multiple places

## Additional Notes

- The frontend is built correctly and has all the necessary code
- The backend is deployed and working perfectly
- The only issue is the missing connection configuration
- This is a configuration issue, not a code issue

## Expected Behavior After Fix

When working correctly, you should see in browser console:

```
=== GETTING RECOMMENDATIONS ===
Session ID: abc-123-def
API URL configured: true
Actual API URL: https://recs-api-...-uc.a.run.app
Webhook secret configured: true/false
Checking API health at: https://recs-api-...-uc.a.run.app/health
Health check successful: {ok: true, status: "ready_full", device: "cpu", ...}
Triggering recommendation generation...
Calling recommendation API for session: abc-123-def
Response status: 200
✅ Recommendation generation successful
Recommendation API response: {session_id: "abc-123-def", phase: 2, svd_top_saved: 10, p5_top_saved: 10, ...}
```

And in the database:
```sql
SELECT session_id, model, COUNT(*) as count
FROM recommendations
GROUP BY session_id, model;

-- Should show:
-- abc-123-def | svd | 10
-- abc-123-def | p5  | 10
```
