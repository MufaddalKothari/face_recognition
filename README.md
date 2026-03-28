# Face Search Desktop App – Setup Guide (Python 3.11)

This guide covers **macOS, Windows, and Linux** installation, virtual environment setup, and running the app.

---

# 🚀 1. Install Python 3.11

## 🍎 macOS

### Option A (Recommended – Homebrew)

```bash
brew update
brew install python@3.11
```

### Verify

```bash
python3.11 --version
```

---

### Option B (Official Installer)

1. Go to: https://www.python.org/downloads/
2. Download **Python 3.11**
3. Run installer
4. Ensure:

   * ✅ “Add Python to PATH” (important)

---

## 🪟 Windows

### Option A (Recommended – Official Installer)

1. Download Python 3.11 from:
   https://www.python.org/downloads/

2. Run installer and **CHECK**:

   ```
   ✔ Add Python to PATH
   ```

3. Verify in Command Prompt:

```bash
python --version
```

or

```bash
python3.11 --version
```

---

### Option B (winget)

```bash
winget install Python.Python.3.11
```

---

## 🐧 Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev -y
```

Verify:

```bash
python3.11 --version
```

---

# 📦 2. Create Virtual Environment

Navigate to your project folder:

```bash
cd /path/to/your/project
```

Create venv:

```bash
python3.11 -m venv face_env
```

---

# ⚡ 3. Activate Virtual Environment

## macOS / Linux

```bash
source face_env/bin/activate
```

## Windows (Command Prompt)

```bash
face_env\Scripts\activate
```

## Windows (PowerShell)

```bash
face_env\Scripts\Activate.ps1
```

---

# 📥 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# ▶️ 5. Run the App

```bash
python ui.py
```

---

# 🧠 Notes

* First run will **download InsightFace models automatically**
* Models are stored in:

  ```
  ~/.insightface/
  ```
* App runs fully on **CPU (no GPU required)**

---

# ❌ Deactivate Virtual Environment

```bash
deactivate
```

---

# 🔧 Troubleshooting

## macOS: Command not found (python3.11)

```bash
brew link python@3.11 --force
```

---

## Windows: Execution policy error (PowerShell)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Linux: Missing build tools

```bash
sudo apt install build-essential python3.11-dev
```

---

# ✅ Quick Start (All Platforms)

```bash
python3.11 -m venv face_env
source face_env/bin/activate   # or Windows equivalent
pip install -r requirements.txt
python ui.py
```

---

You're ready to go 🚀
