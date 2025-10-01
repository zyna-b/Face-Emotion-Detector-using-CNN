# Deployment Guide for Streamlit Community Cloud

Follow these steps to deploy your Face Emotion Detector app on Streamlit Community Cloud:

## Prerequisites

- ✅ Git repository initialized (Done!)
- ✅ All files committed (Done!)
- GitHub account
- Streamlit Community Cloud account

## Step 1: Push to GitHub

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Repository name: `face-emotion-detector` (or your preferred name)
   - Keep it **Public** (required for free Streamlit deployment)
   - **Do NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. **Connect your local repository to GitHub:**

   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/face-emotion-detector.git
   git branch -M main
   git push -u origin main
   ```

   Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 2: Deploy on Streamlit Community Cloud

1. **Sign up for Streamlit Community Cloud:**
   - Go to https://share.streamlit.io/
   - Click "Sign up" and authenticate with your GitHub account

2. **Create a new app:**
   - Click "New app" button
   - Select your GitHub repository: `YOUR_USERNAME/face-emotion-detector`
   - **Main file path:** `app.py`
   - **Branch:** `main`
   - **App URL:** Choose a custom URL (e.g., `face-emotion-detector`)
   - Click "Deploy!"

3. **Wait for deployment:**
   - Streamlit will automatically install dependencies from `requirements.txt`
   - This may take 3-5 minutes for the first deployment
   - You'll see the build logs in real-time

## Step 3: Verify Deployment

Once deployed, your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

## Troubleshooting

### If deployment fails:

1. **Check build logs** in the Streamlit Cloud dashboard
2. **Common issues:**
   - Large file size: The model file `fer2013_emotion_cnn.h5` is large. If it exceeds GitHub's 100MB limit:
     - Use [Git LFS](https://git-lfs.github.com/) for large files
     - Or host the model externally and download it at runtime

### For large model files (>100MB):

If your model file exceeds GitHub's limit, add Git LFS:

```bash
# Install Git LFS
git lfs install

# Track the model file
git lfs track "*.h5"
git add .gitattributes

# Commit and push
git add fer2013_emotion_cnn.h5
git commit -m "Add model with Git LFS"
git push
```

## Updating Your App

After making changes:

```bash
git add .
git commit -m "Description of changes"
git push
```

Streamlit will automatically redeploy your app!

## App Settings (Optional)

In Streamlit Cloud dashboard, you can:
- Set custom domain
- Configure secrets (if needed)
- View analytics
- Restart the app
- View logs

## Current Configuration

✅ Python version: 3.9 (specified in `.python-version`)
✅ Dependencies: Listed in `requirements.txt`
✅ System packages: Listed in `packages.txt` (for TensorFlow dependencies)
✅ Streamlit config: Custom theme in `.streamlit/config.toml`

## Important Files for Deployment

- `app.py` - Main application file
- `requirements.txt` - Python dependencies
- `packages.txt` - System-level dependencies
- `.python-version` - Python version specification
- `fer2013_emotion_cnn.h5` - Trained model (large file!)
- `.streamlit/config.toml` - Streamlit configuration
- `.gitignore` - Files to exclude from Git

## Need Help?

- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud
- Streamlit Community Forum: https://discuss.streamlit.io/
- GitHub Issues: Open an issue in your repository

---

Happy Deploying! 🚀
