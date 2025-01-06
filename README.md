# CPGE Ranking App

**Streamlit application for comparing and ranking CPGE (Classes Préparatoires aux Grandes Écoles) based on your academic profile and various performance indicators.**

> *Découvrez quelles CPGE correspondent le mieux à votre profil en fonction de vos notes, de la sélectivité et des taux de réussite. Comparez les établissements, estimez vos chances d’admission et optimisez votre stratégie de classement pour maximiser vos opportunités.*

---

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Demo](#demo)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Contributing](#contributing)
8. [License](#license)

---

## Overview

This application helps students aiming to join a CPGE program in France to:
- **Evaluate** their chances of admission based on their academic profile (notes, weights, etc.).
- **Compare** CPGE institutions using several metrics: acceptance probability, selectivity, success rates, and more.
- **Visualize** the results in a dynamic ranking table and an interactive map of France.
- **Optimize** their ranking strategy according to different weighting profiles (for example: *Réaliste, Équilibré, Ambitieux*).

---

## Key Features

- **Customizable Weights**: Adjust how much you care about acceptance likelihood versus selectivity and success rates.
- **Preset Profiles**: Quickly set “Réaliste,” “Équilibré,” or “Ambitieux” weighting schemes with a single click.
- **Detailed Ranking**: Filter, rank, and preview CPGE programs (MPSI, PCSI, ECG, etc.) based on your inputs.
- **Interactive Map**: View top CPGE institutions on a geographical map, with hover information on each school’s metrics.
- **Automatic Score Calculation**: Combines your own notes, acceptance probabilities, and institution data into one composite *SSS* (Scoring Support System).

---

## Demo

A live version of the app is available here:
- **Streamlit Community Cloud**: [**Demo Link**](https://streamlit.io/cloud) 

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/your-username/your-cpge-ranking-repo.git
   ```

2. **Navigate** into the project folder:
  ```bash
  cd your-cpge-ranking-repo
  ```

3. **Install** the required Python packages (adjust filename if needed):
  ```bash
  pip install -r requirements.txt
```

---

## Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. Open the URL displayed in your terminal (typically http://localhost:8501) to access the application.

3. Enter your personal data in the notes and weights section, adjust the SSS weights or select a preset button, then calculate rankings to see your results.

---

## Project structure

```bash
your-cpge-ranking-repo/
├── data/
│   └── joined_file_mp_pc_ecg.csv   # CSV with CPGE data
├── cpge_ranking.py                          # Main Streamlit application
├── requirements.txt                # Dependencies list
├── README.md                       # This README
```




