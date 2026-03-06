# 📋 GitHub Repository Setup Instructions

## Step-by-Step Guide to Create Your MARIN Framework Repository

---

## STEP 1: Create GitHub Account (If Needed)

1. Go to: https://github.com
2. Click "Sign Up"
3. Create account with your email

---

## STEP 2: Create New Repository

1. Log in to GitHub
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in:
   - **Repository name:** `MARIN-Framework`
   - **Description:** `Multiplex Adaptive Reinforcement Intervention Network for Real-Time Misinformation Containment`
   - **Visibility:** ✅ Public
   - **Initialize:** ❌ Do NOT check any boxes (no README, no .gitignore, no license)
4. Click **"Create repository"**

---

## STEP 3: Upload Files

### Option A: Upload via GitHub Web Interface (Easiest)

1. On your new empty repository page, click **"uploading an existing file"**
2. Drag and drop ALL files from the `MARIN-Framework` folder
3. Commit message: `Initial commit: MARIN Framework v1.0`
4. Click **"Commit changes"**

### Option B: Upload via Git Command Line

```bash
# Navigate to the MARIN-Framework folder
cd MARIN-Framework

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: MARIN Framework v1.0"

# Add remote (replace YOUR-USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR-USERNAME/MARIN-Framework.git

# Push to GitHub
git push -u origin main
```

---

## STEP 4: Verify Upload

Check that your repository has this structure:

```
MARIN-Framework/
├── README.md                 ✅
├── LICENSE                   ✅
├── requirements.txt          ✅
├── src/
│   ├── __init__.py          ✅
│   ├── marin_network.py     ✅
│   └── marin_agent.py       ✅
├── configs/
│   └── default_config.yaml  ✅
├── data/
│   └── README.md            ✅
├── docs/
│   ├── equations.md         ✅
│   └── graphical_abstract.png ✅
└── results/                  ✅
```

---

## STEP 5: Get Your Repository URL

Your repository URL will be:
```
https://github.com/YOUR-USERNAME/MARIN-Framework
```

**Copy this URL!** You'll need it for the manuscript.

---

## STEP 6: Update Manuscript

Replace the placeholder URL in your manuscript's **"Data and Code Availability"** section:

**Change FROM:**
```
https://github.com/[repository-to-be-created]/MARIN-Framework
```

**Change TO:**
```
https://github.com/YOUR-USERNAME/MARIN-Framework
```

---

## STEP 7: Add Repository Badge to README (Optional)

After uploading, you can add status badges. Your README already includes:

```markdown
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)]
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]
```

---

## 📁 Files Included in This Package

| File | Description | Size |
|------|-------------|------|
| `README.md` | Main repository documentation | ~5 KB |
| `LICENSE` | MIT License | ~1 KB |
| `requirements.txt` | Python dependencies | ~1 KB |
| `src/__init__.py` | Package initialization | ~0.5 KB |
| `src/marin_network.py` | Multiplex network model | ~12 KB |
| `src/marin_agent.py` | DDQN agent implementation | ~15 KB |
| `configs/default_config.yaml` | Hyperparameters | ~5 KB |
| `data/README.md` | Data documentation | ~1 KB |
| `docs/equations.md` | Mathematical formulations | ~4 KB |
| `docs/graphical_abstract.png` | Visual summary | ~460 KB |

---

## ❓ Troubleshooting

### Problem: "Repository already exists"
**Solution:** Choose a different name or delete the existing repo

### Problem: "Permission denied"
**Solution:** Make sure you're logged in to the correct GitHub account

### Problem: "File too large"
**Solution:** GitHub has a 100MB limit. The graphical abstract (~460KB) is fine.

---

## 📧 Need Help?

If you encounter any issues, feel free to contact:
- Anjali Bhadre: anjalibhadre38@gmail.com
- Harshvardhan Ghongade: ghongade@gmail.com

---

*Created for MARIN Framework - NERCCS/NEJCS 2026*
