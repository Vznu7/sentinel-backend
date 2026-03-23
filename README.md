# CardioScan Backend (Render)

This repository contains backend-only files for Render deployment.

## Required files
- api.py
- requirements.txt
- render.yaml
- ct_config.json
- model.pkl

## Deploy on Render
1. Create a new Render Blueprint or Web Service from this repository.
2. Build command: pip install -r requirements.txt
3. Start command: python api.py
4. Health check path: /api/health

## Notes
- Do not commit local runtime database or scan data.
- For hardware worker integration, use server_base in ct_config.json to point app.py worker to Render backend URL.
