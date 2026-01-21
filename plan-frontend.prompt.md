Plan: Build Frontend for Review Radar
Create a modern web UI that uploads a review CSV to the backend, shows analysis results (sentiment, keywords, aspects, failures, top products), downloads generated files, and triggers AI summaries—using the existing Flask API (CORS on, no auth).

Steps
Scaffold frontend (Next.js/React) with pages/components for upload, results dashboard, downloads, and AI summary; configure backend base URL via env.
Implement CSV upload form (multipart file field) calling POST /analyze app.py:115-360; show progress, handle 400/500/504 errors.
Render analysis results from /analyze response: sentiment counts, positive/negative keywords, aspects, failures, top products; show charts using returned PNGs via GET /outputs/<file> app.py:92-113.
Add download buttons for all generated files (predictions.csv, keywords, aspect, failures, products, comparison metrics/report, PNGs) fetched from /outputs/....
Add AI summary action: POST /formulate (JSON, llm_provider + optional model) app.py:362-531; display markdown; surface 400 when predictions absent or provider misconfigured.
Add model comparison view: GET /compare app.py:533-776; render metrics/report and link to saved outputs in /outputs/.

Further Considerations
Backend URL selection: Option A hardcode Render URL; Option B env-driven for local vs prod.
Visualizations: Option A reuse backend PNGs; Option B render client-side charts from response data.
Timeouts/UX: Show 300s max spinner for /analyze; offer cancel/retry if 504.
