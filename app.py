"""
Knowledge Gap Detection & Personalized Learning Recommendation System
Flask Backend Application Server - Integrated with Machine Learning Model Pipeline
"""

import os
import json
import random
import warnings
import urllib.parse
import math
import numpy as np
import pandas as pd
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, make_response

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "app", "templates"),
    static_folder=os.path.join(BASE_DIR, "app", "static")
)
app.secret_key = 'knowledge_gap_ai_secret_key_2026'

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

MODEL = None
LABEL_ENCODER = None
LSTM_MODEL = None
GRU_MODEL = None
TRANSFORMER_MODEL = None
AUTOENCODER_MODEL = None
GNN_MODEL = None
GLOBAL_RESULT_CACHE = {}

try:
    import joblib
    model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
    encoder_path = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
    
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
        print("✔ Successfully loaded XGBoost Model best_model.pkl")
    if os.path.exists(encoder_path):
        LABEL_ENCODER = joblib.load(encoder_path)
        print("✔ Successfully loaded Label Encoder label_encoder.pkl")
except Exception as e:
    print(f"⚠️ Note: Joblib loading info: {e}")

# Load All 5 Advanced Deep Learning & Knowledge Tracing Architectures
try:
    import torch
    import torch.nn as nn
    import math

    # 1. PyTorch LSTM DKT
    class PyTorchLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=7, hidden_size=64, batch_first=True)
            self.dropout = nn.Dropout(0.3)
            self.fc1 = nn.Linear(64, 32)
            self.fc2 = nn.Linear(32, 3)
            self.relu = nn.ReLU()

        def forward(self, x):
            out, _ = self.lstm(x)
            x = out[:, -1, :]
            x = self.dropout(self.relu(self.fc1(x)))
            return self.fc2(x)

    # 2. PyTorch GRU Model
    class PyTorchGRU(nn.Module):
        def __init__(self, input_size=7, hidden_size=32, num_layers=1, num_classes=3):
            super().__init__()
            self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    # 3. Transformer DKT (Self-Attentive Knowledge Tracing - SAKT)
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model=32, max_len=10):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TransformerDKT(nn.Module):
        def __init__(self, input_size=7, d_model=32, nhead=2, num_layers=1, num_classes=3):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=10)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, num_classes)
            
        def forward(self, x):
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            out = self.transformer_encoder(x)
            return self.fc(out[:, -1, :])

    # 4. Knowledge Autoencoder
    class KnowledgeAutoencoder(nn.Module):
        def __init__(self, input_dim=7, latent_dim=4):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, input_dim)
            )
            
        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent

    # Load Model Weights
    lstm_path = os.path.join(BASE_DIR, "models", "lstm_model.pth")
    if os.path.exists(lstm_path):
        LSTM_MODEL = PyTorchLSTM()
        LSTM_MODEL.load_state_dict(torch.load(lstm_path, weights_only=True))
        LSTM_MODEL.eval()
        print("✔ Successfully loaded PyTorch LSTM Deep Knowledge Tracing Model lstm_model.pth")

    gru_path = os.path.join(BASE_DIR, "models", "gru_model.pth")
    if os.path.exists(gru_path):
        GRU_MODEL = PyTorchGRU()
        GRU_MODEL.load_state_dict(torch.load(gru_path, weights_only=True))
        GRU_MODEL.eval()
        print("✔ Successfully loaded PyTorch GRU Model gru_model.pth")

    trans_path = os.path.join(BASE_DIR, "models", "transformer_dkt_model.pth")
    if os.path.exists(trans_path):
        TRANSFORMER_MODEL = TransformerDKT()
        TRANSFORMER_MODEL.load_state_dict(torch.load(trans_path, weights_only=True), strict=False)
        TRANSFORMER_MODEL.eval()
        print("✔ Successfully loaded Transformer-DKT (SAKT) Model transformer_dkt_model.pth")

    ae_path = os.path.join(BASE_DIR, "models", "autoencoder_model.pth")
    if os.path.exists(ae_path):
        AUTOENCODER_MODEL = KnowledgeAutoencoder(input_dim=7, latent_dim=4)
        AUTOENCODER_MODEL.load_state_dict(torch.load(ae_path, weights_only=True))
        AUTOENCODER_MODEL.eval()
        print("✔ Successfully loaded Knowledge Autoencoder Model autoencoder_model.pth")

except Exception as e:
    print(f"⚠️ PyTorch Deep Learning Model Loading Info: {e}")

# EXTREMELY DETAILED COMPREHENSIVE ACADEMIC MASTER STUDY NOTEBOOKS DATABASE
DETAILED_MASTER_NOTES = {
    "Circle Graph": {
        "title": "Circle Graph (Pie Chart) & Data Visualization Comprehensive Master Notebook",
        "overview": "A Circle Graph (or Pie Chart) is a circular statistical graphic divided into proportional slices to illustrate numerical proportions, percentages, and categorical data distributions across a given population.",
        "theory": "The entire circle represents a total of 100% or a central angle of 360 degrees. Each sector's central angle is directly proportional to the relative frequency of the data category it represents: Central Angle = (Category Value / Total Value) × 360°. Pie charts are ideal for displaying discrete nominal or ordinal categorical distributions where parts sum to a single unified whole.",
        "formulas": [
            "Central Angle Formula: θ = (Category Frequency / Total Population) × 360°",
            "Percentage Formula: P% = (Category Frequency / Total Population) × 100%",
            "Category Frequency Reconstruction: f = (θ / 360°) × Total Population",
            "Sector Arc Length: s = (θ / 360°) × 2 π r",
            "Sector Area: Area = (θ / 360°) × π r²"
        ],
        "examples": [
            {
                "level": "Level 1: Basic Sector Percentage & Angle",
                "problem": "In a school of 400 students, 100 students choose Mathematics as their favorite subject. Compute the central angle for Mathematics on a pie chart.",
                "step1": "Find the category fraction: 100 / 400 = 0.25 (or 25%).",
                "step2": "Multiply by 360°: θ = 0.25 × 360° = 90°.",
                "step3": "Verify percentage: 0.25 × 100% = 25%.",
                "solution": "The central angle for Mathematics is 90° (representing 25% of the pie chart)."
            },
            {
                "level": "Level 2: Reconstructing Population Values from Angle",
                "problem": "A pie chart showing monthly expenses has a housing sector of 144°. If total monthly expenses are $5,000, find the exact dollar amount spent on housing.",
                "step1": "Apply Reconstruction Formula: f = (144° / 360°) × $5,000.",
                "step2": "Simplify angle fraction: 144 / 360 = 0.40 (40%).",
                "step3": "Compute amount: 0.40 × $5,000 = $2,000.",
                "step4": "Verify remaining budget: $5,000 - $2,000 = $3,000 (60% or 216°).",
                "solution": "The monthly housing expense is exactly $2,000."
            },
            {
                "level": "Level 3: Multi-Category Comparative Distribution Analysis",
                "problem": "A company surveys 500 workers: 200 drive, 150 train, 100 bus, 50 walk. Find percentage & central angle for every category and verify total completeness.",
                "step1": "Drive: (200 / 500) = 40% -> Angle = 0.40 × 360° = 144°.",
                "step2": "Train: (150 / 500) = 30% -> Angle = 0.30 × 360° = 108°.",
                "step3": "Bus: (100 / 500) = 20% -> Angle = 0.20 × 360° = 72°.",
                "step4": "Walk: (50 / 500) = 10% -> Angle = 0.10 × 360° = 36°.",
                "solution": "Sum Check: 144° + 108° + 72° + 36° = 360° (Verified 100% Complete)."
            },
            {
                "level": "Level 4: Advanced Multi-Chart Comparative Proportion",
                "problem": "Company A (1,000 employees) has a 30% IT sector (108°). Company B (2,500 employees) has a 20% IT sector (72°). Compare total IT headcount.",
                "step1": "Company A IT Headcount: 0.30 × 1,000 = 300 employees.",
                "step2": "Company B IT Headcount: 0.20 × 2,500 = 500 employees.",
                "step3": "Compare: Company B has 200 MORE IT employees than Company A despite having a smaller central angle (72° vs 108°).",
                "solution": "Company B IT headcount (500) > Company A IT headcount (300)."
            }
        ],
        "pitfalls": [
            "Confusing central angles (sum to 360°) with percentages (sum to 100%). Never label a 90° angle as 90%!",
            "Attempting to draw pie charts with negative numbers, continuous time-series, or non-additive data.",
            "Failing to verify that the sum of all calculated central angles equals exactly 360°.",
            "Misinterpreting slice sizes when comparing two pie charts with unequal total sample sizes.",
            "Using 3D tilted pie chart perspectives that visually distort sector proportions."
        ],
        "tips": "Always perform a sum check! Central angles MUST sum to 360° and percentages MUST sum to 100%. Use a protractor aligned at the exact center vertex when constructing circle graphs."
    },
    "Reading a Ruler": {
        "title": "Reading a Ruler & Metric Measurement Comprehensive Master Notebook",
        "overview": "Accurate linear measurement using metric scale rulers is a fundamental skill across physics, engineering, architectural drafting, and manufacturing quality control.",
        "theory": "The metric ruler uses base-10 graduations. Main lines mark centimeters (cm) and minor subdivisions mark millimeters (mm), where 1 cm = 10 mm = 0.01 m = 0.001 km. The International System of Units (SI) defines distance using the meter.",
        "formulas": [
            "Metric Unit Conversion: 1 m = 100 cm = 1,000 mm",
            "Centimeter to Millimeter: Length (mm) = Length (cm) × 10",
            "Millimeter to Centimeter: Length (cm) = Length (mm) / 10",
            "Measurement Reading = Major Division (cm) + (Minor Marks × 0.1 cm)",
            "Offset Reading Rule: True Length = End Mark - Start Mark",
            "Measurement Uncertainty = ± 0.5 × Smallest Scale Division"
        ],
        "examples": [
            {
                "level": "Level 1: Basic Reading Conversion",
                "problem": "An object spans 7 major cm marks plus 4 minor mm ticks. Express its length in cm and mm.",
                "step1": "Identify major reading: 7 cm.",
                "step2": "Add minor ticks: 4 mm = 0.4 cm.",
                "step3": "Combine: 7 + 0.4 = 7.4 cm.",
                "solution": "Total length = 7.4 cm or 74 mm."
            },
            {
                "level": "Level 2: Offset Zero Line Measurement",
                "problem": "An object starts at 2.3 cm and ends at 11.8 cm on a ruler. Find its true length.",
                "step1": "Apply Offset Formula: True Length = End Mark - Start Mark.",
                "step2": "Substitute values: 11.8 cm - 2.3 cm.",
                "step3": "Compute: 9.5 cm.",
                "solution": "True length = 9.5 cm (95 mm)."
            },
            {
                "level": "Level 3: Measurement Tolerance & Relative Uncertainty",
                "problem": "A steel rod is measured as 50 mm using a ruler with smallest division 1 mm (uncertainty ±0.5 mm). Calculate percentage uncertainty.",
                "step1": "Absolute Uncertainty = ± 0.5 mm.",
                "step2": "Relative Uncertainty = (0.5 / 50) × 100%.",
                "step3": "Compute: 0.01 × 100% = 1.0%.",
                "solution": "Measurement = 50 mm ± 1.0%."
            },
            {
                "level": "Level 4: Multi-Segment Perimeter Accumulation",
                "problem": "A triangular plate has side 1 = 4.2 cm, side 2 = 58 mm, and side 3 = 0.065 m. Compute total perimeter in cm.",
                "step1": "Convert side 2 to cm: 58 mm / 10 = 5.8 cm.",
                "step2": "Convert side 3 to cm: 0.065 m × 100 = 6.5 cm.",
                "step3": "Sum all sides: 4.2 + 5.8 + 6.5 = 16.5 cm.",
                "solution": "Total perimeter = 16.5 cm."
            }
        ],
        "pitfalls": [
            "Aligning object with physical ruler edge instead of the zero mark line!",
            "Miscounting minor millimeter tick marks (confusing 0.5 cm mid-length ticks with 1 cm marks).",
            "Forgetting that 1 cm = 10 mm (not 100 mm).",
            "Parallax errors caused by looking at ruler tick marks from an angled perspective.",
            "Failing to convert mixed measurement units (cm, mm, m) before performing addition."
        ],
        "tips": "Look straight down at graduation marks perpendicular to the ruler surface to eliminate parallax errors!"
    },
    "Equivalent Fractions": {
        "title": "Equivalent Fractions & Rational Numbers Comprehensive Master Notebook",
        "overview": "Equivalent fractions represent equal numerical values or proportions despite having different numerators and denominators. Essential across algebra, probability, and ratio analysis.",
        "theory": "Based on Multiplicative Identity (a/b) × (k/k) = (a·k)/(b·k) for k ≠ 0. Fractions are simplified by dividing numerator and denominator by their Greatest Common Divisor (GCD). Two fractions a/b and c/d are strictly equivalent if and only if cross-products a·d = b·c.",
        "formulas": [
            "Equivalence Multiplication: (a / b) = (a × k) / (b × k) for k ≠ 0",
            "Simplification Division: (a / b) = (a ÷ d) / (b ÷ d) where d = GCD(a, b)",
            "Cross-Multiplication Test: (a / b) = (c / d) iff (a × d) = (b × c)",
            "Decimal Equivalency: (a / b) = a ÷ b",
            "Rational Difference Test: (a / b) - (c / d) = 0"
        ],
        "examples": [
            {
                "level": "Level 1: Building Equivalent Fraction",
                "problem": "Find an equivalent fraction for 3/7 with denominator 35.",
                "step1": "Determine required multiplier: 35 ÷ 7 = 5.",
                "step2": "Multiply numerator and denominator by 5: (3 × 5) / (7 × 5).",
                "step3": "Compute result: 15 / 35.",
                "solution": "3/7 = 15/35."
            },
            {
                "level": "Level 2: Simplest Form via GCD",
                "problem": "Reduce fraction 48/72 to simplest terms.",
                "step1": "Find GCD(48, 72) = 24.",
                "step2": "Divide numerator & denominator by 24: (48÷24) / (72÷24).",
                "step3": "Compute: 2 / 3.",
                "solution": "Simplest irreducible fraction = 2/3."
            },
            {
                "level": "Level 3: Cross-Multiplication Equivalence Verification",
                "problem": "Check if 14/21 and 26/39 are equivalent.",
                "step1": "Compute cross products: 14 × 39 = 546; 21 × 26 = 546.",
                "step2": "Compare: 546 == 546.",
                "solution": "546 == 546, so fractions are EQUIVALENT."
            },
            {
                "level": "Level 4: Algebraic Equivalent Fraction Equation",
                "problem": "Solve for x in equivalent fractions: (2x + 1) / 5 = 15 / 25.",
                "step1": "Simplify right side: 15/25 = 3/5.",
                "step2": "Equate numerators: 2x + 1 = 3.",
                "step3": "Solve for x: 2x = 2 -> x = 1.",
                "solution": "x = 1."
            }
        ],
        "pitfalls": [
            "Adding or subtracting the same number to numerator and denominator instead of multiplying! Example: (3+2)/(4+2) = 5/6 ≠ 3/4.",
            "Multiplying numerator and denominator by different factors.",
            "Forgetting to check if a fraction can be simplified further using GCD.",
            "Confusing reciprocal fractions with equivalent fractions.",
            "Dividing by zero when attempting to build equivalent fractions."
        ],
        "tips": "Use cross-multiplication (a·d = b·c) for instant equivalence verification on exams!"
    },
    "Finding Percents": {
        "title": "Finding Percents & Ratio Proportions Comprehensive Master Notebook",
        "overview": "Percentage represents a ratio or fraction expressed out of 100 (per centum). Essential for financial interest rates, growth rate modeling, discount rates, and error margins.",
        "theory": "Mathematically P% = P / 100. Any percentage calculation involves three variables: Part (Amount), Whole (Base), and Percent (Rate). Governing equation: Part = (Percent / 100) × Whole.",
        "formulas": [
            "Part Formula: Part = (Percent / 100) × Whole",
            "Percent Formula: Percent (%) = (Part / Whole) × 100%",
            "Whole Formula: Whole = Part / (Percent / 100)",
            "Percentage Change: (%) = [ (New Value - Original) / Original ] × 100%",
            "Compound Growth: Amount = Principal × (1 + r/100)^t"
        ],
        "examples": [
            {
                "level": "Level 1: Percentage of a Number",
                "problem": "Calculate 18% of $450.",
                "step1": "Convert percent to decimal: 18% = 0.18.",
                "step2": "Multiply: 0.18 × 450 = 81.",
                "solution": "18% of $450 is $81."
            },
            {
                "level": "Level 2: Percentage Increase",
                "problem": "Textbook price increased from $80 to $100. Calculate percentage increase.",
                "step1": "Absolute change = $100 - $80 = $20.",
                "step2": "Divide by original: 20 / 80 = 0.25.",
                "step3": "Multiply by 100%: 0.25 × 100% = 25%.",
                "solution": "Price increased by 25%."
            },
            {
                "level": "Level 3: Reverse Percentage",
                "problem": "A discounted item costs $680 after a 15% discount. Find original price.",
                "step1": "Discounted rate = 100% - 15% = 85% = 0.85.",
                "step2": "Original Price = 680 / 0.85 = 800.",
                "solution": "Original price was $800."
            },
            {
                "level": "Level 4: Consecutive Compound Percentage",
                "problem": "A stocks price increases by 20% on Monday, then decreases by 20% on Tuesday. Find net percentage change.",
                "step1": "Start price = $100. Monday price = 100 × 1.20 = $120.",
                "step2": "Tuesday price = 120 × (1 - 0.20) = 120 × 0.80 = $96.",
                "step3": "Net change = (96 - 100) / 100 × 100% = -4%.",
                "solution": "Net change is a 4% DECREASE (Not 0%)."
            }
        ],
        "pitfalls": [
            "Dividing by NEW price instead of ORIGINAL price during percentage change calculations!",
            "Assuming +X% followed by -X% returns to original value.",
            "Forgetting to divide percent by 100 before multiplying.",
            "Confusing percentage points with percentage change.",
            "Misapplying sales tax or discount to incorrect subtotal amounts."
        ],
        "tips": "Mental Math Trick: X% of Y = Y% of X! Example: 16% of 50 = 50% of 16 = 8."
    },
    "Median": {
        "title": "Median & Central Tendency Comprehensive Master Notebook",
        "overview": "The median represents the exact middle numerical value of an ordered statistical dataset. Unlike mean, median is resistant to extreme outliers and skewed distributions.",
        "theory": "Splits ordered dataset into two equal halves. For odd N, it is the single middle item at position (N+1)/2. For even N, it is the mean of middle items at N/2 and (N/2)+1.",
        "formulas": [
            "Position (Odd N): Position = (N + 1) / 2",
            "Position (Even N): Positions (N/2) and (N/2)+1",
            "Median Value (Even N): Median = [ Value(N/2) + Value(N/2 + 1) ] / 2",
            "Grouped Data Median: L + [ ((N/2) - CF) / f ] × h",
            "Interquartile Range: IQR = Q3 - Q1"
        ],
        "examples": [
            {
                "level": "Level 1: Odd Dataset",
                "problem": "Find median of test scores: 85, 92, 67, 74, 88, 95, 71.",
                "step1": "Sort: [67, 71, 74, 85, 88, 92, 95]. N = 7.",
                "step2": "Middle Position = (7+1)/2 = 4th item.",
                "step3": "4th item = 85.",
                "solution": "Median score is 85."
            },
            {
                "level": "Level 2: Even Dataset with Outliers",
                "problem": "Find median of values: 42, 45, 46, 48, 50, 52, 55, 120.",
                "step1": "Data sorted. N = 8. Middle items: 4th (48) & 5th (50).",
                "step2": "Average: (48 + 50) / 2 = 49.",
                "solution": "Median is 49 (outlier 120 does not distort median)."
            },
            {
                "level": "Level 3: Frequency Distribution Median",
                "problem": "Quiz scores out of 5: Score 1 (2), Score 2 (5), Score 3 (8), Score 4 (4), Score 5 (1). Find median.",
                "step1": "Total N = 20 (Even). Middle items: 10th and 11th.",
                "step2": "Cumulative Frequencies: S1(2), S2(7), S3(15). Both 10th & 11th items fall in Score 3.",
                "solution": "Median quiz score is 3."
            },
            {
                "level": "Level 4: Median Shift via Constant Addition & Scaling",
                "problem": "If dataset median is 40, find new median if every number is multiplied by 2 then increased by 5.",
                "step1": "Apply linear transformation: New Median = 2(Original Median) + 5.",
                "step2": "Compute: 2(40) + 5 = 85.",
                "solution": "New median = 85."
            }
        ],
        "pitfalls": [
            "Finding middle element WITHOUT sorting dataset first!",
            "Picking only one middle number for even N datasets.",
            "Confusing Median (middle) with Mean (average) or Mode (most frequent).",
            "Assuming outlier values affect the median as heavily as the mean.",
            "Incorrectly calculating cumulative frequency in grouped data tables."
        ],
        "tips": "Always sort data in ascending order first before crossing off elements from both ends!"
    },
    "Proportion": {
        "title": "Proportions & Ratio Analysis Comprehensive Master Notebook",
        "overview": "A proportion states that two ratios are equal: a/b = c/d. Governs scaling, unit conversions, geometric similarity, and direct/inverse variation.",
        "theory": "Cross-Product Property: a · d = b · c. Direct variation follows y = kx; Inverse variation follows y = k / x.",
        "formulas": [
            "Proportion Equality: (a / b) = (c / d)",
            "Cross-Product Property: a × d = b × c",
            "Direct Variation: y = k × x",
            "Inverse Variation: y = k / x",
            "Scale Factor: Scale = Model Length / Actual Length"
        ],
        "examples": [
            {
                "level": "Level 1: Solving Unknown",
                "problem": "Solve for x: 5 / 8 = x / 48.",
                "step1": "Cross-multiply: 5 × 48 = 8 × x -> 240 = 8x.",
                "step2": "Divide by 8: x = 30.",
                "solution": "x = 30."
            },
            {
                "level": "Level 2: Scale Conversion",
                "problem": "Scale 0.5 in = 4 ft. Blueprint room length = 3.5 in. Find actual feet.",
                "step1": "0.5 / 4 = 3.5 / X.",
                "step2": "0.5X = 14 -> X = 28 ft.",
                "solution": "Actual room length = 28 feet."
            },
            {
                "level": "Level 3: Inverse Variation Worker Problem",
                "problem": "6 workers finish a job in 10 days. How long will 15 workers take?",
                "step1": "Inverse formula: W₁ × D₁ = W₂ × D₂.",
                "step2": "6 × 10 = 15 × D₂ -> 60 = 15 D₂.",
                "solution": "D₂ = 4 days."
            },
            {
                "level": "Level 4: Compound Proportion",
                "problem": "4 machines produce 200 items in 5 hours. How many items will 8 machines produce in 10 hours?",
                "step1": "Rate per machine-hour = 200 / (4 × 20) = 200 / 20 = 10 items/mach-hr.",
                "step2": "Compute for 8 machines in 10 hours: 8 × 10 × 10 = 800 items.",
                "solution": "Output = 800 items."
            }
        ],
        "pitfalls": [
            "Cross-multiplying fractions that are being added or subtracted instead of equated!",
            "Treating inverse variation as direct proportion.",
            "Failing to align units across numerators and denominators.",
            "Forgetting to check for zero denominators.",
            "Assuming scale factor applies linearly to 2D area (Area scales by k²!)."
        ],
        "tips": "Always align units across numerators and denominators!"
    },
    "Quadratic Formula": {
        "title": "Quadratic Formula & Roots Comprehensive Master Notebook",
        "overview": "Universal solution for roots of ax² + bx + c = 0 (a ≠ 0). Governed by Discriminant Δ = b² - 4ac.",
        "theory": "Derived by completing the square. Roots x = (-b ± √(b² - 4ac)) / (2a). Δ > 0 gives 2 real roots; Δ = 0 gives 1 repeated root; Δ < 0 gives complex conjugate roots.",
        "formulas": [
            "Standard Form: a x² + b x + c = 0  (a ≠ 0)",
            "Quadratic Formula: x = [ -b ± √( b² - 4 a c ) ] / ( 2 a )",
            "Discriminant: Δ = b² - 4 a c",
            "Vieta's Formulas: x₁ + x₂ = -b/a,  x₁ × x₂ = c/a",
            "Vertex Coordinates: h = -b / (2a), k = c - (b² / 4a)"
        ],
        "examples": [
            {
                "level": "Level 1: Two Real Roots",
                "problem": "Solve x² - 5x + 6 = 0.",
                "step1": "a = 1, b = -5, c = 6. Δ = 25 - 24 = 1.",
                "step2": "x = (5 ± 1) / 2 -> x = 3 or x = 2.",
                "solution": "Roots are x = 3 and x = 2."
            },
            {
                "level": "Level 2: Repeated Real Root",
                "problem": "Solve 4x² - 12x + 9 = 0.",
                "step1": "a = 4, b = -12, c = 9. Δ = 144 - 144 = 0.",
                "step2": "x = 12 / 8 = 1.5.",
                "solution": "Single root x = 1.5."
            },
            {
                "level": "Level 3: Complex Conjugate Roots",
                "problem": "Solve x² - 4x + 13 = 0.",
                "step1": "a = 1, b = -4, c = 13. Δ = 16 - 52 = -36.",
                "step2": "x = (4 ± 6i) / 2 = 2 ± 3i.",
                "solution": "Roots are x = 2 + 3i and x = 2 - 3i."
            },
            {
                "level": "Level 4: Physics Projectile Maximum Height",
                "problem": "Height h(t) = -5t² + 20t + 15. Find time t when object hits ground (h=0).",
                "step1": "-5t² + 20t + 15 = 0 -> Divide by -5: t² - 4t - 3 = 0.",
                "step2": "a=1, b=-4, c=-3. Δ = 16 - 4(1)(-3) = 28.",
                "step3": "t = (4 + √28) / 2 = (4 + 5.29) / 2 = 4.65 seconds.",
                "solution": "Impact time = 4.65 seconds."
            }
        ],
        "pitfalls": [
            "Forgetting standard form ax² + bx + c = 0 before identifying coefficients!",
            "Sign error with -b when b is negative.",
            "Dividing only the radical term by 2a instead of the entire numerator.",
            "Forgetting imaginary unit i when discriminant is negative.",
            "Confusing vertex max/min point with root x-intercepts."
        ],
        "tips": "Use Vieta's formulas to check: Sum of roots MUST equal -b/a!"
    },
    "Multiplication Whole Numbers": {
        "title": "Multiplication of Whole Numbers Comprehensive Master Notebook",
        "overview": "Binary operation representing repeated addition across equal-sized groups.",
        "theory": "Obeys Commutative, Associative, and Distributive properties over addition.",
        "formulas": [
            "Repeated Addition: a × b = b + b + ... + b (a times)",
            "Distributive Law: a × (b + c) = (a × b) + (a × c)",
            "Commutative Law: a × b = b × a",
            "Associative Law: (a × b) × c = a × (b × c)",
            "Multiplicative Identity: a × 1 = a, Zero: a × 0 = 0"
        ],
        "examples": [
            {
                "level": "Level 1: Mental Math",
                "problem": "Calculate 14 × 25 using distributive property.",
                "step1": "(10 + 4) × 25 = 250 + 100 = 350.",
                "solution": "Product = 350."
            },
            {
                "level": "Level 2: Standard Multi-Digit Algorithm",
                "problem": "Calculate 348 × 46.",
                "step1": "348 × 6 = 2,088.",
                "step2": "348 × 40 = 13,920.",
                "step3": "Sum: 2,088 + 13,920 = 16,008.",
                "solution": "Product = 16,008."
            }
        ],
        "pitfalls": [
            "Forgetting placeholder zero when shifting to tens column in multi-digit multiplication!",
            "Carrying numbers incorrectly during column addition.",
            "Confusing zero multiplication (a × 0 = 0) with identity multiplication (a × 1 = a)."
        ],
        "tips": "Estimate products first to catch magnitude errors!"
    },
    "Computation with Real Numbers": {
        "title": "Computation with Real Numbers & PEMDAS Comprehensive Master Notebook",
        "overview": "Real number calculations governed by strict operator precedence rules.",
        "theory": "PEMDAS / BODMAS hierarchy: Parentheses, Exponents, Multiplication & Division (left to right), Addition & Subtraction (left to right).",
        "formulas": [
            "PEMDAS Hierarchy: P -> E -> MD -> AS",
            "Signed Rules: (-) × (-) = (+), (-) × (+) = (-)",
            "Absolute Value: |-x| = x",
            "Exponent Radical Rule: x^(m/n) = n-th root of (x^m)",
            "Distributive Law: a(b + c) = ab + ac"
        ],
        "examples": [
            {
                "level": "Level 1: PEMDAS Order",
                "problem": "Evaluate 8 + 2 × (3² - 5).",
                "step1": "Parentheses & Exponent: 3² = 9 -> (9 - 5) = 4.",
                "step2": "Multiply: 2 × 4 = 8. Add: 8 + 8 = 16.",
                "solution": "Result = 16."
            },
            {
                "level": "Level 2: Complex Expression",
                "problem": "Simplify: [ (-2)³ + 4 × (5 - 8) ] / [ √49 - (3 - 6)² ].",
                "step1": "Numerator: -8 + 4(-3) = -20.",
                "step2": "Denominator: 7 - (-3)² = 7 - 9 = -2.",
                "step3": "Division: -20 / -2 = 10.",
                "solution": "Result = +10."
            }
        ],
        "pitfalls": [
            "Adding before multiplying in expressions like 5 + 3 × 2!",
            "Confusing -3² (-9) with (-3)² (+9).",
            "Evaluating multiplication before division when division appears to the left."
        ],
        "tips": "Work left to right for operators of equal rank!"
    },
    "Systems of Linear Equations": {
        "title": "Systems of Linear Equations Comprehensive Master Notebook",
        "overview": "Simultaneous equations solved for shared variable values.",
        "theory": "Geometric intersection of lines: Single point (unique solution), Parallel lines (no solution), Coincident lines (infinite solutions).",
        "formulas": [
            "System: a₁x + b₁y = c₁ and a₂x + b₂y = c₂",
            "Cramer's Rule Determinant: D = a₁b₂ - a₂b₁",
            "Cramer's Solutions: x = Dx / D, y = Dy / D",
            "Slope Comparison: m₁ = -a₁/b₁, m₂ = -a₂/b₂"
        ],
        "examples": [
            {
                "level": "Level 1: Elimination Method",
                "problem": "Solve: 2x + 3y = 13 and 4x - 3y = 11.",
                "step1": "Add equations: 6x = 24 -> x = 4.",
                "step2": "Substitute x=4: 2(4) + 3y = 13 -> 3y = 5 -> y = 5/3.",
                "solution": "Solution = (4, 5/3)."
            },
            {
                "level": "Level 2: Cramer's Rule",
                "problem": "Solve using Cramer's rule: 3x + 2y = 7 and 5x + 4y = 13.",
                "step1": "D = (3)(4) - (5)(2) = 12 - 10 = 2.",
                "step2": "Dx = (7)(4) - (13)(2) = 28 - 26 = 2 -> x = 2/2 = 1.",
                "step3": "Dy = (3)(13) - (5)(7) = 39 - 35 = 4 -> y = 4/2 = 2.",
                "solution": "Solution = (1, 2)."
            }
        ],
        "pitfalls": [
            "Sign distribution errors during algebraic elimination!",
            "Confusing D=0 no solution (parallel) with D=0 infinite solutions (coincident)."
        ],
        "tips": "Plug solutions back into BOTH original equations to confirm!"
    },
    "Pythagorean Theorem": {
        "title": "Pythagorean Theorem & Right Triangle Geometry Comprehensive Master Notebook",
        "overview": "Fundamental geometric relationship linking the three sides of any Euclidean right-angled triangle.",
        "theory": "In any right triangle with hypotenuse c and legs a and b, the square of the hypotenuse equals the sum of the squares of the two legs: a² + b² = c².",
        "formulas": [
            "Pythagorean Formula: a² + b² = c²",
            "Hypotenuse: c = √(a² + b²)",
            "Leg Length: a = √(c² - b²),  b = √(c² - a²)",
            "Common Pythagorean Triples: (3,4,5), (5,12,13), (8,15,17), (7,24,25)",
            "Distance in 2D Plane: d = √[ (x₂ - x₁)² + (y₂ - y₁)² ]"
        ],
        "examples": [
            {
                "level": "Level 1: Finding Hypotenuse",
                "problem": "A right triangle has legs a = 6 cm and b = 8 cm. Compute hypotenuse c.",
                "step1": "c² = 6² + 8² = 36 + 64 = 100.",
                "step2": "c = √100 = 10 cm.",
                "solution": "Hypotenuse = 10 cm."
            },
            {
                "level": "Level 2: Ladder Against Wall",
                "problem": "A 13 ft ladder reaches a wall from a base 5 ft away. Find reach height.",
                "step1": "h = √(13² - 5²) = √(169 - 25) = √144 = 12 ft.",
                "solution": "Height = 12 feet."
            },
            {
                "level": "Level 3: 3D Space Diagonal",
                "problem": "Find 3D diagonal of rectangular box with dimensions 3m × 4m × 12m.",
                "step1": "d = √(3² + 4² + 12²) = √(9 + 16 + 144) = √169 = 13 m.",
                "solution": "3D Diagonal = 13 meters."
            }
        ],
        "pitfalls": [
            "Applying Pythagorean Theorem to non-right triangles!",
            "Confusing hypotenuse with shorter leg sides.",
            "Adding side lengths directly instead of squaring them first."
        ],
        "tips": "Recognize Pythagorean triples (3,4,5) and (5,12,13) for lightning-fast test calculations!"
    },
    "Probability and Statistics": {
        "title": "Probability & Statistical Foundations Comprehensive Master Notebook",
        "overview": "Probability measures event likelihood (0 ≤ P(E) ≤ 1), while statistics analyzes data distributions, means, and variances.",
        "theory": "P(E) = Favorable Outcomes / Total Outcomes. Independent events satisfy P(A ∩ B) = P(A) × P(B).",
        "formulas": [
            "Theoretical Probability: P(E) = n(E) / n(S)",
            "Complement Rule: P(E') = 1 - P(E)",
            "Addition Rule: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)",
            "Independent Multiplication: P(A ∩ B) = P(A) × P(B)",
            "Expected Value: E(X) = Σ [ x × P(x) ]"
        ],
        "examples": [
            {
                "level": "Level 1: Single Event",
                "problem": "Bag has 5 red, 3 blue, 2 green marbles. Find P(Blue).",
                "step1": "Total n(S) = 10. Favorable n(E) = 3.",
                "solution": "P(Blue) = 3/10 = 0.30 (30%)."
            },
            {
                "level": "Level 2: Compound Independent",
                "problem": "Coin flipped twice. Find P(Two Heads).",
                "step1": "P(H) × P(H) = (1/2) × (1/2) = 1/4.",
                "solution": "P(HH) = 1/4 = 0.25 (25%)."
            }
        ],
        "pitfalls": [
            "Adding probabilities when events are independent instead of multiplying!",
            "Forgetting to subtract intersection in general addition rule."
        ],
        "tips": "Probabilities MUST always fall between 0.0 and 1.0!"
    },
    "Exponents and Radicals": {
        "title": "Exponents & Radical Operations Comprehensive Master Notebook",
        "overview": "Exponents represent repeated multiplication, while radicals represent fractional power inverse operations.",
        "theory": "Exponent rules govern power simplification. Fractional exponent x^(a/b) represents b-th root of x^a.",
        "formulas": [
            "Product Rule: x^a × x^b = x^(a + b)",
            "Quotient Rule: x^a / x^b = x^(a - b)",
            "Power Rule: (x^a)^b = x^(a × b)",
            "Negative Exponent: x^(-a) = 1 / x^a",
            "Fractional Exponent: x^(a/b) = b-th root of (x^a)"
        ],
        "examples": [
            {
                "level": "Level 1: Power & Product Rules",
                "problem": "Simplify (2x³)² × x⁴.",
                "step1": "(2x³)² = 4x⁶. 4x⁶ × x⁴ = 4x¹⁰.",
                "solution": "Simplified = 4x¹⁰."
            },
            {
                "level": "Level 2: Fractional Exponents",
                "problem": "Evaluate 27^(2/3).",
                "step1": "∛27 = 3. 3² = 9.",
                "solution": "27^(2/3) = 9."
            }
        ],
        "pitfalls": [
            "Adding bases instead of powers! Writing x² × x³ = (2x)⁵ instead of x⁵.",
            "Confusing negative exponents x⁻² = 1/x² with negative values."
        ],
        "tips": "Remember non-zero x⁰ = 1!"
    },
    "Perimeter and Area": {
        "title": "Perimeter & Area Geometry Comprehensive Master Notebook",
        "overview": "Perimeter measures outer boundary length; Area measures 2D enclosed surface space.",
        "theory": "Calculated using geometric formulas based on shape classification.",
        "formulas": [
            "Rectangle: Perimeter = 2(L + W), Area = L × W",
            "Triangle: Perimeter = a + b + c, Area = (1/2) × Base × Height",
            "Circle: Circumference = 2 π r, Area = π r²",
            "Trapezoid Area: Area = (1/2) × (b₁ + b₂) × Height",
            "Square: Perimeter = 4s, Area = s²"
        ],
        "examples": [
            {
                "level": "Level 1: Circle Area",
                "problem": "Circle radius r = 7 cm. Find Area (π ≈ 22/7).",
                "step1": "Area = (22/7) × 49 = 154 cm².",
                "solution": "Area = 154 cm²."
            },
            {
                "level": "Level 2: Triangle Area",
                "problem": "Triangle base = 12 m, height = 9 m. Find area.",
                "step1": "Area = (1/2) × 12 × 9 = 54 m².",
                "solution": "Area = 54 m²."
            }
        ],
        "pitfalls": [
            "Using diameter instead of radius when calculating circle area π r²!",
            "Confusing linear units (cm) with square area units (cm²)."
        ],
        "tips": "Square units for area (m²), linear units for perimeter!"
    },
    "Linear Functions and Slope": {
        "title": "Linear Functions & Slope Algebra Comprehensive Master Notebook",
        "overview": "Linear functions define constant rate of change relationships graphed as straight lines.",
        "theory": "Slope m represents steepness (rise over run). Parallel lines have equal slopes; perpendicular lines have negative reciprocal slopes.",
        "formulas": [
            "Slope Formula: m = (y₂ - y₁) / (x₂ - x₁)",
            "Slope-Intercept Form: y = m x + b",
            "Point-Slope Form: y - y₁ = m (x - x₁)",
            "Standard Form: A x + B y = C",
            "Perpendicular Rule: m_perp = -1 / m"
        ],
        "examples": [
            {
                "level": "Level 1: Slope Between Points",
                "problem": "Find slope between (2, 5) and (6, 17).",
                "step1": "m = (17 - 5) / (6 - 2) = 12 / 4 = 3.",
                "solution": "Slope m = 3."
            },
            {
                "level": "Level 2: Line Equation",
                "problem": "Find line equation with slope m = 4 through (3, 2).",
                "step1": "y - 2 = 4(x - 3) -> y = 4x - 10.",
                "solution": "y = 4x - 10."
            }
        ],
        "pitfalls": [
            "Reversing x and y in slope formula (Δx / Δy instead of Δy / Δx)!",
            "Sign mistakes when subtracting negative coordinates."
        ],
        "tips": "Vertical lines have UNDEFINED slope, Horizontal lines have ZERO slope!"
    },
    "Polynomial Operations": {
        "title": "Polynomial Operations & Algebra Comprehensive Master Notebook",
        "overview": "Polynomials are algebraic expressions built from variables, exponents, and coefficients.",
        "theory": "Addition/subtraction combine like terms. Multiplication uses distributive FOIL expansion.",
        "formulas": [
            "FOIL Expansion: (a + b)(c + d) = ac + ad + bc + bd",
            "Difference of Squares: a² - b² = (a + b)(a - b)",
            "Square of Binomial: (a + b)² = a² + 2ab + b²",
            "Cube of Binomial: (a + b)³ = a³ + 3a²b + 3ab² + b³"
        ],
        "examples": [
            {
                "level": "Level 1: FOIL Binomial",
                "problem": "Expand (3x + 4)(2x - 5).",
                "step1": "6x² - 15x + 8x - 20 = 6x² - 7x - 20.",
                "solution": "Expanded = 6x² - 7x - 20."
            }
        ],
        "pitfalls": [
            "Writing (a + b)² = a² + b² (Forgetting the middle 2ab term!)",
            "Sign distribution errors when subtracting polynomials."
        ],
        "tips": "Distribute negative signs to EVERY term inside parentheses when subtracting!"
    },
    "Trigonometric Ratios": {
        "title": "Trigonometric Ratios & Right Triangles Comprehensive Master Notebook",
        "overview": "Trigonometric ratios relate angles of a right triangle to ratios of its side lengths (SOH CAH TOA).",
        "theory": "Defined for acute angle θ: Sine = Opp/Hyp, Cosine = Adj/Hyp, Tangent = Opp/Adj. Pythagorean Identity: sin²θ + cos²θ = 1.",
        "formulas": [
            "Sine Ratio: sin(θ) = Opposite / Hypotenuse",
            "Cosine Ratio: cos(θ) = Adjacent / Hypotenuse",
            "Tangent Ratio: tan(θ) = Opposite / Adjacent",
            "Pythagorean Identity: sin²(θ) + cos²(θ) = 1",
            "Reciprocal Ratios: csc(θ) = 1/sin(θ), sec(θ) = 1/cos(θ)"
        ],
        "examples": [
            {
                "level": "Level 1: Basic Ratios",
                "problem": "Triangle opposite=3, adjacent=4, hypotenuse=5. Find sin(θ) & cos(θ).",
                "step1": "sin(θ) = 3/5 = 0.60, cos(θ) = 4/5 = 0.80.",
                "solution": "sin(θ) = 0.60, cos(θ) = 0.80."
            }
        ],
        "pitfalls": [
            "Confusing Opposite and Adjacent sides relative to angle θ!",
            "Using Degree mode when Radians are specified."
        ],
        "tips": "Remember SOH CAH TOA for instant recall!"
    },
    "Logarithmic Functions": {
        "title": "Logarithmic Functions & Exponential Inverse Comprehensive Master Notebook",
        "overview": "Logarithms are inverse operations to exponentiation: log_b(x) = y means b^y = x.",
        "theory": "Base 10 = log(x), Base e = ln(x). Logarithm rules convert products and powers into additions and multiplications.",
        "formulas": [
            "Definition: log_b(x) = y <=> b^y = x",
            "Product Law: log_b(x × y) = log_b(x) + log_b(y)",
            "Quotient Law: log_b(x / y) = log_b(x) - log_b(y)",
            "Power Law: log_b(x^k) = k × log_b(x)",
            "Change of Base: log_b(x) = log_c(x) / log_c(b)"
        ],
        "examples": [
            {
                "level": "Level 1: Basic Evaluation",
                "problem": "Evaluate log₂(64).",
                "step1": "2⁶ = 64.",
                "solution": "log₂(64) = 6."
            },
            {
                "level": "Level 2: Equation Solving",
                "problem": "Solve log₃(x) + log₃(9) = 4.",
                "step1": "log₃(9x) = 4 -> 9x = 3⁴ = 81 -> x = 9.",
                "solution": "x = 9."
            }
        ],
        "pitfalls": [
            "Taking logarithm of zero or negative numbers (x > 0 required)!",
            "Writing log(x + y) = log(x) + log(y) (Incorrect!)."
        ],
        "tips": "Always verify argument x > 0 for all logarithms!"
    },
    "Matrices and Determinants": {
        "title": "Matrices & Determinants Comprehensive Master Notebook",
        "overview": "Matrices are rectangular arrays of numbers used to represent linear transformations and systems.",
        "theory": "Matrix multiplication is non-commutative (A·B ≠ B·A). For 2x2 matrix [[a,b],[c,d]], Determinant det(A) = ad - bc.",
        "formulas": [
            "2x2 Determinant: det(A) = (a × d) - (b × c)",
            "2x2 Inverse Matrix: A⁻¹ = (1 / det(A)) × [[d, -b], [-c, a]]",
            "Matrix Multiplication: Entry (i,j) = Row i · Column j dot product"
        ],
        "examples": [
            {
                "level": "Level 1: 2x2 Determinant",
                "problem": "Compute det of A = [[4, 2], [3, 5]].",
                "step1": "det(A) = (4 × 5) - (2 × 3) = 20 - 6 = 14.",
                "solution": "det(A) = 14."
            }
        ],
        "pitfalls": [
            "Assuming matrix multiplication is commutative (A·B = B·A is FALSE!)",
            "Attempting to invert a singular matrix where det(A) = 0."
        ],
        "tips": "If det(A) = 0, the matrix has NO inverse!"
    },
    "Coordinate Geometry and Distance": {
        "title": "Coordinate Geometry & Distance Formula Comprehensive Master Notebook",
        "overview": "Analytic geometry mapping shapes onto Cartesian coordinate planes using algebraic equations.",
        "theory": "Distance formula derived from Pythagorean Theorem on coordinate grid. Midpoint calculates average coordinates.",
        "formulas": [
            "Distance Formula: d = √[ (x₂ - x₁)² + (y₂ - y₁)² ]",
            "Midpoint Formula: M = ( (x₁ + x₂)/2 , (y₁ + y₂)/2 )",
            "Circle Equation: (x - h)² + (y - k)² = r²",
            "Section Formula: P = ( (m x₂ + n x₁)/(m+n) , (m y₂ + n y₁)/(m+n) )"
        ],
        "examples": [
            {
                "level": "Level 1: Distance Between Points",
                "problem": "Find distance between P1(1, 2) and P2(4, 6).",
                "step1": "Δx = 3, Δy = 4.",
                "step2": "d = √(3² + 4²) = √25 = 5.",
                "solution": "Distance = 5 units."
            },
            {
                "level": "Level 2: Midpoint Calculation",
                "problem": "Find midpoint of segment with endpoints (-2, 8) and (6, 4).",
                "step1": "Mx = (-2 + 6) / 2 = 4 / 2 = 2.",
                "step2": "My = (8 + 4) / 2 = 12 / 2 = 6.",
                "solution": "Midpoint = (2, 6)."
            }
        ],
        "pitfalls": [
            "Forgetting to square coordinate differences before adding in distance formula!",
            "Subtracting x1 from y1 instead of x2 from x1."
        ],
        "tips": "Distance is ALWAYS non-negative!"
    }
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'student_name' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def get_student_key():
    reg = str(session.get('register_number', '')).strip()
    name = str(session.get('student_name', '')).strip()
    if reg:
        return reg.lower()
    if name:
        return name.lower()
    return "global_default_student"

def load_questions(filter_skill=None):
    """Load questions from recommendation/questions.json"""
    q_file = os.path.join(BASE_DIR, "recommendation", "questions.json")
    if os.path.exists(q_file):
        with open(q_file, "r") as f:
            data = json.load(f)
            for idx, item in enumerate(data):
                item["id"] = int(item.get("id", idx + 1))
                s_name = str(item.get("skill", item.get("skill_name", "General Skill"))).strip()
                item["skill_name"] = s_name
                item["skill"] = s_name

            if filter_skill:
                filtered = [item for item in data if item.get("skill_name", "").strip().lower() == filter_skill.strip().lower()]
                if filtered:
                    return filtered
            return data
    return []

SKILL_VIDEO_MAPPING = {
    "Circle Graph": {
        "embed_url": "https://www.youtube.com/embed/gTshbBhhV3s",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Circle+Graph",
        "edu_url": "https://www.youtube.com/results?search_query=Circle+Graph+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Circle+Graph+math+tutorial"
    },
    "Reading a Ruler": {
        "embed_url": "https://www.youtube.com/embed/7u__k0K1eGg",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Reading+a+Ruler",
        "edu_url": "https://www.youtube.com/results?search_query=Reading+a+ruler+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Reading+a+ruler+math+tutorial"
    },
    "Equivalent Fractions": {
        "embed_url": "https://www.youtube.com/embed/qcHHhd6HizI",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Equivalent+Fractions",
        "edu_url": "https://www.youtube.com/results?search_query=Equivalent+fractions+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Equivalent+fractions+math+tutorial"
    },
    "Finding Percents": {
        "embed_url": "https://www.youtube.com/embed/rR95Cbcjuis",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Finding+Percents",
        "edu_url": "https://www.youtube.com/results?search_query=Finding+percents+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Finding+percents+math+tutorial"
    },
    "Median": {
        "embed_url": "https://www.youtube.com/embed/B1HEzNTGeZ4",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Median",
        "edu_url": "https://www.youtube.com/results?search_query=Median+statistics+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Median+statistics+math+tutorial"
    },
    "Proportion": {
        "embed_url": "https://www.youtube.com/embed/GO56OiUjzp0",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Proportion",
        "edu_url": "https://www.youtube.com/results?search_query=Proportions+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Proportions+math+tutorial"
    },
    "Quadratic Formula": {
        "embed_url": "https://www.youtube.com/embed/i7idZfS8t8w",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Quadratic+Formula",
        "edu_url": "https://www.youtube.com/results?search_query=Quadratic+formula+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Quadratic+formula+math+tutorial"
    },
    "Multiplication Whole Numbers": {
        "embed_url": "https://www.youtube.com/embed/RVYwunbpMHA",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Multiplication",
        "edu_url": "https://www.youtube.com/results?search_query=Multiplication+whole+numbers+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Multiplication+whole+numbers+math+tutorial"
    },
    "Computation with Real Numbers": {
        "embed_url": "https://www.youtube.com/embed/d6vhko_Wf3g",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Real+Numbers",
        "edu_url": "https://www.youtube.com/results?search_query=Computation+with+real+numbers+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Computation+with+real+numbers+math+tutorial"
    },
    "Systems of Linear Equations": {
        "embed_url": "https://www.youtube.com/embed/bAerID24QJ0",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Systems+of+Linear+Equations",
        "edu_url": "https://www.youtube.com/results?search_query=Solving+systems+of+linear+equations+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Systems+of+linear+equations+math+tutorial"
    },
    "Pythagorean Theorem": {
        "embed_url": "https://www.youtube.com/embed/AA6RfgP-AHU",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Pythagorean+Theorem",
        "edu_url": "https://www.youtube.com/results?search_query=Pythagorean+theorem+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Pythagorean+theorem+math+tutorial"
    },
    "Probability and Statistics": {
        "embed_url": "https://www.youtube.com/embed/UzxYlbK2c7E",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Probability",
        "edu_url": "https://www.youtube.com/results?search_query=Probability+and+statistics+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Probability+and+statistics+math+tutorial"
    },
    "Exponents and Radicals": {
        "embed_url": "https://www.youtube.com/embed/LwCRRUa8yTU",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Exponents+and+Radicals",
        "edu_url": "https://www.youtube.com/results?search_query=Exponents+and+radicals+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Exponents+and+radicals+math+tutorial"
    },
    "Perimeter and Area": {
        "embed_url": "https://www.youtube.com/embed/rSVmrpuo03c",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Perimeter+and+Area",
        "edu_url": "https://www.youtube.com/results?search_query=Perimeter+and+area+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Perimeter+and+area+math+tutorial"
    },
    "Linear Functions and Slope": {
        "embed_url": "https://www.youtube.com/embed/ADloWn0gEjs",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Linear+Functions",
        "edu_url": "https://www.youtube.com/results?search_query=Linear+functions+and+slope+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Linear+functions+and+slope+math+tutorial"
    },
    "Polynomial Operations": {
        "embed_url": "https://www.youtube.com/embed/ffLLmV4mZwU",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Polynomial+Operations",
        "edu_url": "https://www.youtube.com/results?search_query=Polynomial+operations+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Polynomial+operations+math+tutorial"
    },
    "Trigonometric Ratios": {
        "embed_url": "https://www.youtube.com/embed/PUB0TaZ7bhA",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Trigonometry",
        "edu_url": "https://www.youtube.com/results?search_query=Trigonometric+ratios+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Trigonometric+ratios+math+tutorial"
    },
    "Logarithmic Functions": {
        "embed_url": "https://www.youtube.com/embed/kqVpPSjY2xM",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Logarithmic+Functions",
        "edu_url": "https://www.youtube.com/results?search_query=Logarithmic+functions+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Logarithmic+functions+math+tutorial"
    },
    "Matrices and Determinants": {
        "embed_url": "https://www.youtube.com/embed/2uB23YnL0F0",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Matrices",
        "edu_url": "https://www.youtube.com/results?search_query=Matrices+and+determinants+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Matrices+and+determinants+math+tutorial"
    },
    "Coordinate Geometry and Distance": {
        "embed_url": "https://www.youtube.com/embed/0X8aK26N06k",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Coordinate+Geometry",
        "edu_url": "https://www.youtube.com/results?search_query=Coordinate+geometry+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Coordinate+geometry+math+tutorial"
    }
}

def load_resources():
    """Load recommendations for ALL 10 skills with detailed academic master notes & verified video streams"""
    r_file = os.path.join(BASE_DIR, "recommendation", "resources.csv")
    recs = []
    
    student_key = get_student_key()
    result_data = session.get('result_data') or GLOBAL_RESULT_CACHE.get(student_key) or GLOBAL_RESULT_CACHE.get('latest_submission', {})
    
    weak_skills = result_data.get('weak_skills', [])
    strengths = result_data.get('strengths', [])
    
    student_scores = {}
    tested_keys = set()

    for item in weak_skills + strengths:
        name = str(item.get('name', '')).strip()
        if name:
            key = name.lower().replace(" ", "").replace("-", "")
            student_scores[key] = float(item.get('score', 0.0))
            tested_keys.add(key)
        
    if os.path.exists(r_file):
        df = pd.read_csv(r_file)
        temp_list = []

        for idx, row in df.iterrows():
            skill = str(row.get("skill_name", "Skill")).strip()
            key = skill.lower().replace(" ", "").replace("-", "")
            
            is_tested = key in tested_keys
            
            # STRICT FILTER: Only show recommendations for topics ATTENDED in student assessment!
            if tested_keys and not is_tested:
                continue

            score_pct = student_scores.get(key, 100.0)

            if is_tested and score_pct < 70.0:
                category = "critical"
                badge_text = f"🚨 Critical Focus Area (Score: {score_pct}%)"
                badge_class = "badge-danger-custom"
                status_text = "Knowledge Gap - Remediation Needed"
            else:
                category = "mastered"
                badge_text = f"🌟 Mastered Skill (Score: {score_pct}%)"
                badge_class = "badge-success-custom"
                status_text = "Skill Mastered - Practice to Retain"

            encoded_skill = urllib.parse.quote_plus(skill)
            
            v_info = SKILL_VIDEO_MAPPING.get(skill, {
                "embed_url": "https://www.youtube.com/embed/bAerID24QJ0",
                "khan_url": f"https://www.khanacademy.org/search?referer=%2F&page_search_query={encoded_skill}",
                "edu_url": f"https://www.youtube.com/results?search_query={encoded_skill}+educational+math+lesson",
                "web_url": f"https://www.google.com/search?tbm=vid&q={encoded_skill}+math+lesson+tutorial"
            })
            
            khan_video_url = v_info.get("khan_url", f"https://www.khanacademy.org/search?referer=%2F&page_search_query={encoded_skill}")
            edu_video_url = v_info.get("edu_url", f"https://www.youtube.com/results?search_query={encoded_skill}+educational+math+lesson")
            web_video_url = v_info.get("web_url", f"https://www.google.com/search?tbm=vid&q={encoded_skill}+math+lesson+tutorial")

            detailed_notes = DETAILED_MASTER_NOTES.get(skill, {
                "title": f"{skill} Master Academic Study Notebook",
                "overview": f"Comprehensive study overview and conceptual analysis for {skill}.",
                "theory": f"Theoretical foundation and mathematical framework governing {skill}.",
                "formulas": [f"Standard formula for {skill}"],
                "examples": [
                    {
                        "level": "Level 1: Basic Problem",
                        "problem": f"Standard problem for {skill}.",
                        "step1": "Identify parameters.",
                        "step2": "Apply formula.",
                        "solution": "Solved result."
                    }
                ],
                "pitfalls": [f"Common exam mistake in {skill}."],
                "tips": "Review all key definitions before taking assessment."
            })

            temp_list.append({
                "skill_name": skill,
                "score_pct": score_pct,
                "category": category,
                "badge_text": badge_text,
                "badge_class": badge_class,
                "status_text": status_text,
                "description": f"Detailed academic learning module for {skill} (Current performance: {score_pct}%).",
                "estimated_time": "25 Mins",
                "type": "Video & Master Notes",
                "khan_video_url": khan_video_url,
                "edu_video_url": edu_video_url,
                "video_embed_url": v_info["embed_url"],
                "web_video_url": web_video_url,
                "detailed_notes": detailed_notes
            })

        # Sort with lowest scoring skills FIRST
        recs = sorted(temp_list, key=lambda x: x['score_pct'])
        
    return recs

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/analytics")
@login_required
def analytics():
    return result()

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        student_name = request.form.get('student_name', '').strip()
        register_number = request.form.get('register_number', '').strip()
        department = request.form.get('department', 'Computer Science')
        email = request.form.get('email', '').strip()

        if not student_name or not register_number:
            return render_template("login.html", error="Please enter student name and register number.")

        session['student_name'] = student_name
        session['register_number'] = register_number
        session['department'] = department
        session['email'] = email
        session.pop('result_data', None)
        session.pop('active_quiz_questions', None)
        session.pop('active_quiz_ids', None)

        next_url = request.args.get('next') or url_for('instructions')
        return redirect(next_url)
        
    return render_template("login.html")

@app.route("/login_post", methods=["POST"])
def login_post():
    session['student_name'] = request.form.get('student_name', 'Student Candidate')
    session['register_number'] = request.form.get('register_number', '210101001')
    session['department'] = request.form.get('department', 'Computer Science')
    session['email'] = request.form.get('email', 'student@institution.edu')
    session.pop('result_data', None)
    session.pop('active_quiz_questions', None)
    session.pop('active_quiz_ids', None)
    return redirect(url_for('instructions'))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route("/instructions")
@login_required
def instructions():
    session.pop('active_quiz_ids', None)
    session.pop('active_quiz_questions', None)
    student_name = session.get('student_name', 'Student Candidate')
    register_number = session.get('register_number', '210101001')
    department = session.get('department', 'Computer Science')
    return render_template("instructions.html", student_name=student_name, register_number=register_number, department=department)

@app.route("/new_quiz")
@login_required
def new_quiz():
    session.pop('active_quiz_ids', None)
    session.pop('active_quiz_questions', None)
    return redirect(url_for('quiz'))

@app.route("/quiz")
@login_required
def quiz():
    filter_skill = request.args.get('skill')
    all_questions = load_questions(filter_skill=filter_skill)
    
    if filter_skill:
        questions = all_questions
    else:
        # Group questions by skill and shuffle skill order for dynamic topic sampling
        skill_groups = {}
        for q in all_questions:
            s = q.get('skill_name', q.get('skill', 'General Skill'))
            if s not in skill_groups:
                skill_groups[s] = []
            skill_groups[s].append(q)
            
        all_skills = list(skill_groups.keys())
        random.shuffle(all_skills) # Randomize skill selection on every assessment run!
        
        sampled = []
        for s in all_skills:
            if len(sampled) < 10:
                sampled.append(random.choice(skill_groups[s]))
                
        if len(sampled) < 10:
            remaining = [q for q in all_questions if q not in sampled]
            needed = 10 - len(sampled)
            if remaining:
                sampled.extend(random.sample(remaining, min(needed, len(remaining))))
            
        random.shuffle(sampled) # Randomize question presentation order
        questions = sampled[:10]

    session['active_quiz_ids'] = [q['id'] for q in questions]
    session.pop('active_quiz_questions', None)
    return render_template("quiz.html", questions=questions, active_skill=filter_skill)

@app.route("/submit", methods=["POST"])
@login_required
def submit():
    data = request.get_json() or {}
    user_answers = data.get('answers', {})
    
    active_ids = session.get('active_quiz_ids', [])
    all_questions = load_questions()
    
    if active_ids:
        questions = [q for qid in active_ids for q in all_questions if q['id'] == qid]
    else:
        questions = all_questions[:10]

    skill_stats = {}
    total_correct = 0
    question_review = []

    for idx, q in enumerate(questions):
        q_id_str = str(q['id'])
        q_id_int = int(q['id'])
        skill = q.get('skill_name', q.get('skill', 'General Skill'))
        raw_correct_ans = str(q.get('answer', '')).strip()
        options = q.get('options', [])
        q_text = q.get('question', f"Question #{idx+1}")

        if skill not in skill_stats:
            skill_stats[skill] = {"total": 0, "correct": 0, "wrong": 0}
        
        skill_stats[skill]["total"] += 1
        
        # Look up user answer STRICTLY by question ID (prevents key swapping)
        user_ans_idx = user_answers.get(q_id_str)
        if user_ans_idx is None:
            user_ans_idx = user_answers.get(q_id_int)

        selected_opt_str = "Not Answered"
        is_correct = False

        if user_ans_idx is not None:
            try:
                selected_idx = int(user_ans_idx)
                if 0 <= selected_idx < len(options):
                    selected_opt_str = str(options[selected_idx]).strip()
                    
                    # 1. Exact case-insensitive string match
                    if selected_opt_str.lower() == raw_correct_ans.lower():
                        is_correct = True
                        
                    # 2. Normalized string match (preserve minus sign!)
                    def norm(s):
                        return s.lower().replace("°", "").replace("%", "").replace("$", "").replace(" ", "")
                    
                    if norm(selected_opt_str) == norm(raw_correct_ans):
                        is_correct = True
                        
                    # 3. Numeric option index match if answer is "0", "1", "2", "3"
                    try:
                        if selected_idx == int(raw_correct_ans):
                            is_correct = True
                    except ValueError:
                        pass
                        
                    # 4. Numerical / Fraction math evaluation (e.g. 1/2 == 2/4, 25% == 25, 90° == 90)
                    try:
                        def parse_math(s):
                            clean = s.lower().replace("°", "").replace("%", "").replace("$", "").strip()
                            if "/" in clean:
                                p = clean.split("/")
                                return float(p[0]) / float(p[1])
                            return float(clean)

                        v1 = parse_math(selected_opt_str)
                        v2 = parse_math(raw_correct_ans)
                        if v1 is not None and v2 is not None and abs(v1 - v2) < 1e-5:
                            is_correct = True
                    except Exception:
                        pass
            except Exception as e:
                print(f"Evaluation note: {e}")

        if is_correct:
            skill_stats[skill]["correct"] += 1
            total_correct += 1
        else:
            skill_stats[skill]["wrong"] += 1

        question_review.append({
            "number": idx + 1,
            "question": q_text,
            "skill": skill,
            "selected": selected_opt_str,
            "correct": raw_correct_ans,
            "is_correct": is_correct,
            "options": options
        })

    total_questions = len(questions) if len(questions) > 0 else 10
    accuracy = round((total_correct / total_questions) * 100, 1)
    knowledge_gap_score = round(100.0 - accuracy, 1)
    wrong_count = total_questions - total_correct

    print(f"📊 [Quiz Evaluation] Total Questions: {total_questions} | Correct Answers: {total_correct} | Accuracy: {accuracy}%")

    if knowledge_gap_score > 50:
        gap_level = "High Knowledge Gap"
        gap_badge_class = "badge-danger-custom"
    elif knowledge_gap_score > 25:
        gap_level = "Moderate Knowledge Gap"
        gap_badge_class = "badge-warning-custom"
    else:
        gap_level = "Low Knowledge Gap"
        gap_badge_class = "badge-success-custom"

    if MODEL is not None:
        try:
            attempt_cnt = wrong_count + 1
            hint_cnt = wrong_count
            hint_tot = 3
            sample_features = pd.DataFrame([{
                "correct": 1 if total_correct > (total_questions / 2) else 0,
                "attempt_count": attempt_cnt,
                "hint_count": hint_cnt,
                "hint_ratio": hint_cnt / (hint_tot + 1),
                "attempt_hint_sum": attempt_cnt + hint_cnt,
                "log_ms_response": math.log1p(15000),
                "is_first_action_hint": 1 if hint_cnt > 0 else 0,
                "opportunity": total_questions,
                "position": 1,
                "tutor_mode": "tutor",
                "answer_type": "algebra",
                "type": "Algebra"
            }])
            
            # Filter columns based on model preprocessor expectations
            try:
                expected_num = MODEL.named_steps["preprocessor"].transformers_[0][2]
                expected_cat = MODEL.named_steps["preprocessor"].transformers_[1][2]
                expected_cols = list(expected_num) + list(expected_cat)
                sample_features = sample_features[[c for c in expected_cols if c in sample_features.columns]]
            except Exception:
                pass

            ml_pred = MODEL.predict(sample_features)[0]
            if LABEL_ENCODER is not None:
                ml_label = LABEL_ENCODER.inverse_transform([ml_pred])[0]
                gap_level = f"{ml_label} Knowledge Gap"
                if ml_label == "High":
                    gap_badge_class = "badge-danger-custom"
                elif ml_label == "Medium":
                    gap_badge_class = "badge-warning-custom"
                else:
                    gap_badge_class = "badge-success-custom"
                
                print(f"🤖 [XGBoost ML Output] Class: {ml_pred} -> Prediction: '{ml_label}'")
        except Exception as ml_err:
            print(f"ML Prediction Note: {ml_err}")

    # PyTorch LSTM Deep Knowledge Tracing Model Prediction
    dl_logits_str = "[0.000, 0.000, 0.000]"
    if LSTM_MODEL is not None:
        try:
            import torch
            seq_sample = torch.zeros((1, 10, 7), dtype=torch.float32)
            seq_sample[0, :, 0] = 1.0 if total_correct > (total_questions / 2) else 0.0
            seq_sample[0, :, 1] = float(wrong_count + 1)
            seq_sample[0, :, 2] = float(wrong_count)
            seq_sample[0, :, 3] = float(wrong_count / 4.0)
            seq_sample[0, :, 4] = float(math.log1p(15000))
            seq_sample[0, :, 5] = float(math.log1p(30000))
            seq_sample[0, :, 6] = float(2 * wrong_count + 1)
            
            with torch.no_grad():
                dl_output = LSTM_MODEL(seq_sample)
                dl_pred_class = torch.argmax(dl_output, dim=1).item()
                dl_labels = ["High", "Low", "Medium"]
                dl_label = dl_labels[dl_pred_class] if dl_pred_class < len(dl_labels) else "Medium"
                
                logits_arr = dl_output.numpy()[0]
                dl_logits_str = f"[{logits_arr[0]:.4f}, {logits_arr[1]:.4f}, {logits_arr[2]:.4f}]"
                print(f"🧠 [PyTorch LSTM DKT Output] Logits: {dl_logits_str} -> Class: {dl_pred_class} ('{dl_label}')")

                if GRU_MODEL is not None:
                    gru_out = GRU_MODEL(seq_sample)
                    gru_class = torch.argmax(gru_out, dim=1).item()
                    print(f"⚡ [PyTorch GRU Output] Class: {gru_class} ('{dl_labels[gru_class]}')")

                if TRANSFORMER_MODEL is not None:
                    trans_out = TRANSFORMER_MODEL(seq_sample)
                    trans_class = torch.argmax(trans_out, dim=1).item()
                    print(f"🤖 [Transformer-DKT Output] Class: {trans_class} ('{dl_labels[trans_class]}')")

                if AUTOENCODER_MODEL is not None:
                    _, latent_vec = AUTOENCODER_MODEL(seq_sample[0, 0, :].unsqueeze(0))
                    latent_str = np.array2string(latent_vec.numpy()[0], precision=3)
                    print(f"🔲 [Knowledge Autoencoder Output] 4D Latent Vector: {latent_str}")
        except Exception as dl_err:
            print(f"PyTorch Deep Models Prediction Note: {dl_err}")

    weak_skills = []
    strengths = []
    skill_names = []
    skill_scores = []

    for skill, stats in skill_stats.items():
        pct = round((stats["correct"] / stats["total"]) * 100, 1) if stats["total"] > 0 else 0.0
        wrong_c = stats.get("wrong", 0)
        skill_names.append(skill)
        skill_scores.append(pct)

        item_dict = {
            "name": skill,
            "score": pct,
            "wrong_count": wrong_c,
            "correct_count": stats["correct"],
            "total": stats["total"]
        }

        if pct < 70.0:
            weak_skills.append(item_dict)
        else:
            strengths.append(item_dict)

    # Sort each section by score
    weak_skills = sorted(weak_skills, key=lambda s: s["score"])
    strengths = sorted(strengths, key=lambda s: s["score"], reverse=True)

    res = {
        "student_name": session.get('student_name', 'Student Candidate'),
        "register_number": session.get('register_number', '210101001'),
        "department": session.get('department', 'Computer Science'),
        "score": total_correct,
        "total_questions": total_questions,
        "correct_answers": total_correct,
        "wrong_answers": total_questions - total_correct,
        "unanswered_answers": 0,
        "accuracy": accuracy,
        "knowledge_gap_score": knowledge_gap_score,
        "gap_level": gap_level,
        "gap_badge_class": gap_badge_class,
        "pytorch_logits": dl_logits_str,
        "weak_skills": weak_skills,
        "strengths": strengths,
        "question_review": question_review,
        "skill_names_json": json.dumps(skill_names),
        "skill_scores_json": json.dumps(skill_scores)
    }

    session['result_data'] = res
    s_key = get_student_key()
    GLOBAL_RESULT_CACHE[s_key] = res
    GLOBAL_RESULT_CACHE['latest_submission'] = res

    return jsonify({"status": "success", "redirect": url_for('loading')})

@app.route("/loading")
@login_required
def loading():
    return render_template("loading.html")

@app.route("/result")
@login_required
def result():
    s_key = get_student_key()
    result_data = session.get('result_data') or GLOBAL_RESULT_CACHE.get(s_key) or GLOBAL_RESULT_CACHE.get('latest_submission')
    
    if not result_data:
        tot_q = 10
        result_data = {
            "student_name": session.get('student_name', 'Student Candidate'),
            "register_number": session.get('register_number', '210101001'),
            "department": session.get('department', 'Computer Science'),
            "score": 0,
            "total_questions": tot_q,
            "correct_answers": 0,
            "wrong_answers": tot_q,
            "unanswered_answers": 0,
            "accuracy": 0.0,
            "knowledge_gap_score": 100.0,
            "gap_level": "Assessment Pending",
            "gap_badge_class": "badge-brand",
            "weak_skills": [],
            "strengths": [],
            "question_review": [],
            "skill_names_json": json.dumps([]),
            "skill_scores_json": json.dumps([])
        }
    return render_template("result.html", result_data=result_data)

@app.route("/review")
@login_required
def review():
    s_key = get_student_key()
    result_data = session.get('result_data') or GLOBAL_RESULT_CACHE.get(s_key) or GLOBAL_RESULT_CACHE.get('latest_submission')
    if not result_data:
        return redirect(url_for('quiz'))
    return render_template("review.html", result_data=result_data)

@app.route("/recommendations")
@login_required
def recommendations():
    resources = load_resources()
    return render_template("recommendations.html", recommendations=resources)

@app.route("/pdf_notes/<skill>")
@login_required
def pdf_notes(skill):
    """Generates an authentic, highly detailed, print-ready A4 PDF document layout with zero KeyError risk"""
    notes = DETAILED_MASTER_NOTES.get(skill, {})
    
    overview = notes.get('overview', notes.get('summary', f"Comprehensive academic study overview for {skill}."))
    theory = notes.get('theory', f"Theoretical principles and mathematical framework governing {skill}.")
    formulas = notes.get('formulas', [f"Standard mathematical rule for {skill}."])
    examples = notes.get('examples', [])
    pitfalls = notes.get('pitfalls', [f"Common exam mistake in {skill}."])
    tips = notes.get('tips', "Review all key formulas and definitions before solving exam questions.")
    
    student_name = session.get('student_name', 'Student Candidate')
    reg_number = session.get('register_number', '210101001')
    department = session.get('department', 'Computer Science')

    formulas_html = "".join([f'<div class="formula-box"><code>{f}</code></div>' for f in formulas])
    
    examples_html = ""
    for idx, ex in enumerate(examples, 1):
        steps = ""
        if 'step1' in ex: steps += f"<p><strong>Step 1:</strong> {ex['step1']}</p>"
        if 'step2' in ex: steps += f"<p><strong>Step 2:</strong> {ex['step2']}</p>"
        if 'step3' in ex: steps += f"<p><strong>Step 3:</strong> {ex['step3']}</p>"
        if 'step4' in ex: steps += f"<p><strong>Step 4:</strong> {ex['step4']}</p>"
        
        examples_html += f"""
        <div class="example-card">
            <h4><span class="badge-level">{ex.get('level', f'Example {idx}')}</span></h4>
            <p class="problem-text"><strong>Problem Statement:</strong> {ex.get('problem', '')}</p>
            <div class="solution-steps">
                {steps}
            </div>
            <div class="solution-badge">
                ✔ <strong>Final Answer:</strong> {ex.get('solution', '')}
            </div>
        </div>
        """

    pitfalls_html = "".join([f'<li>{p}</li>' for p in pitfalls])

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{skill} - Official Master Academic PDF Notebook</title>
        <style>
            @page {{
                size: A4;
                margin: 20mm 15mm 20mm 15mm;
            }}
            body {{
                font-family: 'Georgia', 'Times New Roman', serif;
                color: #0f172a;
                line-height: 1.6;
                background: #ffffff;
                margin: 0;
                padding: 0;
            }}
            .header-banner {{
                text-align: center;
                border-bottom: 3px double #4338ca;
                padding-bottom: 15px;
                margin-bottom: 25px;
            }}
            .header-banner h1 {{
                font-family: 'Helvetica Neue', Arial, sans-serif;
                color: #3730a3;
                font-size: 24px;
                margin: 0 0 5px 0;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .header-banner p {{
                font-size: 13px;
                color: #64748b;
                margin: 0;
                font-style: italic;
            }}
            .student-meta {{
                background: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 12px 18px;
                margin-bottom: 25px;
                display: flex;
                justify-content: space-between;
                font-size: 13px;
                font-family: 'Helvetica Neue', Arial, sans-serif;
            }}
            .section {{
                margin-bottom: 30px;
                page-break-inside: avoid;
            }}
            .section-title {{
                font-family: 'Helvetica Neue', Arial, sans-serif;
                color: #1e1b4b;
                font-size: 16px;
                border-bottom: 2px solid #6366f1;
                padding-bottom: 4px;
                margin-bottom: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .content-text {{
                font-size: 14px;
                text-align: justify;
                margin-bottom: 10px;
            }}
            .formula-box {{
                background: #eef2ff;
                border-left: 4px solid #4338ca;
                padding: 10px 15px;
                margin: 8px 0;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
                color: #1e1b4b;
            }}
            .example-card {{
                background: #f0fdf4;
                border: 1px solid #bbf7d0;
                border-radius: 6px;
                padding: 15px;
                margin-bottom: 18px;
                page-break-inside: avoid;
            }}
            .badge-level {{
                background: #166534;
                color: white;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-family: 'Helvetica Neue', Arial, sans-serif;
            }}
            .problem-text {{
                font-size: 14px;
                color: #14532d;
                margin: 8px 0;
            }}
            .solution-steps p {{
                font-size: 13px;
                margin: 4px 0;
                color: #166534;
            }}
            .solution-badge {{
                margin-top: 10px;
                padding: 8px 12px;
                background: #dcfce7;
                border-left: 4px solid #16a34a;
                font-size: 13px;
                font-weight: bold;
                color: #14532d;
            }}
            .pitfall-list {{
                background: #fef2f2;
                border: 1px solid #fecaca;
                border-radius: 6px;
                padding: 15px 15px 15px 35px;
                font-size: 13px;
                color: #991b1b;
            }}
            .tip-box {{
                background: #fffbeb;
                border: 1px solid #fde68a;
                border-radius: 6px;
                padding: 15px;
                font-size: 13px;
                color: #78350f;
                font-weight: bold;
            }}
            .footer {{
                text-align: center;
                font-size: 11px;
                color: #94a3b8;
                border-top: 1px solid #e2e8f0;
                padding-top: 15px;
                margin-top: 40px;
                font-family: 'Helvetica Neue', Arial, sans-serif;
            }}
            .btn-controls {{
                text-align: right;
                margin-bottom: 20px;
            }}
            .btn-print {{
                background: #4338ca;
                color: white;
                border: none;
                padding: 10px 22px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                cursor: pointer;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            @media print {{
                .btn-controls {{ display: none; }}
                body {{ padding: 0; }}
            }}
        </style>
        <script>
            window.onload = function() {{
                setTimeout(function() {{ window.print(); }}, 800);
            }};
        </script>
    </head>
    <body>
        
        <div class="btn-controls">
            <button class="btn-print" onclick="window.print()">🖨️ Save as PDF / Print Document</button>
        </div>

        <div class="header-banner">
            <h1>EduGap AI - Master Academic Notebook</h1>
            <p>Explainable and Prerequisite-Aware Knowledge Gap Detection Using Machine Learning and Deep Knowledge Tracing for Personalized Learning</p>
        </div>

        <div class="student-meta">
            <div><strong>Candidate Name:</strong> {student_name}</div>
            <div><strong>Reg Number:</strong> {reg_number}</div>
            <div><strong>Department:</strong> {department}</div>
            <div><strong>Topic:</strong> {skill}</div>
        </div>

        <!-- Module 1: Executive Overview -->
        <div class="section">
            <div class="section-title">1. Executive Concept Overview</div>
            <div class="content-text">{overview}</div>
            <div class="content-text" style="font-style: italic; color: #475569;"><strong>Theoretical Foundation:</strong> {theory}</div>
        </div>

        <!-- Module 2: Governing Equations -->
        <div class="section">
            <div class="section-title">2. Governing Formulas, Axioms & Theorems</div>
            {formulas_html}
        </div>

        <!-- Module 3: Exhaustive Solved Problems -->
        <div class="section">
            <div class="section-title">3. Step-by-Step Solved Academic Examples</div>
            {examples_html}
        </div>

        <!-- Module 4: Pitfalls -->
        <div class="section">
            <div class="section-title">4. Common Exam Pitfalls & Misconceptions</div>
            <ul class="pitfall-list">
                {pitfalls_html}
            </ul>
        </div>

        <!-- Module 5: Revision Strategy -->
        <div class="section">
            <div class="section-title">5. Pro Exam Revision Strategy</div>
            <div class="tip-box">
                💡 <strong>Exam Pro Tip:</strong> {tips}
            </div>
        </div>

        <div class="footer">
            Page 1 of 1 &bull; Official Academic Document Generated by Knowledge Gap AI System &bull; Confidential
        </div>

    </body>
    </html>
    """
    return make_response(html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)