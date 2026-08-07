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

# Load Deep Knowledge Tracing Model from Notebook 07
try:
    import torch
    import torch.nn as nn

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

    lstm_path = os.path.join(BASE_DIR, "models", "lstm_model.pth")
    if os.path.exists(lstm_path):
        LSTM_MODEL = PyTorchLSTM()
        LSTM_MODEL.load_state_dict(torch.load(lstm_path))
        LSTM_MODEL.eval()
        print("✔ Successfully loaded PyTorch LSTM Deep Knowledge Tracing Model lstm_model.pth")
except Exception as e:
    print(f"⚠️ PyTorch LSTM load info: {e}")

# EXTREMELY DETAILED COMPREHENSIVE ACADEMIC MASTER STUDY NOTEBOOKS DATABASE
DETAILED_MASTER_NOTES = {
    "Circle Graph": {
        "title": "Circle Graph (Pie Chart) Master Academic Study Notebook",
        "overview": "A Circle Graph (commonly known as a Pie Chart) is a circular statistical graphic divided into proportional sectors or slices. Each sector represents a specific numerical proportion of the whole dataset. The total area of the circle corresponds to 100% of the data or a complete 360° central angle rotation around the center point.",
        "theory": "The geometric principle of a Circle Graph relies on direct linear proportionality between categorical frequency and central angle measure. If a dataset contains total frequency N and a specific category has frequency f, the sector angle θ is directly proportional to (f / N). Circle graphs excel at visualizing part-to-whole relationships, budget distributions, and categorical survey breakdowns.",
        "formulas": [
            "Central Angle (θ) = (Category Frequency f / Total Frequency N) × 360°",
            "Percentage (%) = (Category Frequency f / Total Frequency N) × 100%",
            "Category Frequency f = (Central Angle θ / 360°) × Total Frequency N",
            "Sector Arc Length (s) = (θ / 360°) × 2πr",
            "Sector Area (A) = (θ / 360°) × πr²"
        ],
        "examples": [
            {
                "level": "Level 1: Basic Central Angle Calculation",
                "problem": "In a class of 40 students, 10 students chose Mathematics as their favorite subject. Calculate the central angle of the Mathematics sector in a circle graph.",
                "step1": "Identify Given Values: Category Frequency f = 10, Total Frequency N = 40.",
                "step2": "Apply Central Angle Formula: θ = (f / N) × 360°.",
                "step3": "Substitute values: θ = (10 / 40) × 360° = 0.25 × 360° = 90°.",
                "solution": "The central angle for the Mathematics sector is exactly 90° (a right angle)."
            },
            {
                "level": "Level 2: Reconstructing Population Values from Angle",
                "problem": "A pie chart displaying monthly budget expenses shows a housing expenditure sector with a central angle of 144°. If the total monthly budget is $5,000, determine the exact dollar amount spent on housing.",
                "step1": "Identify Given Values: Central Angle θ = 144°, Total Budget N = $5,000.",
                "step2": "Apply Frequency Reconstruction Formula: f = (θ / 360°) × N.",
                "step3": "Calculate fraction: 144° / 360° = 0.40 (or 40%).",
                "step4": "Multiply by total budget: f = 0.40 × $5,000 = $2,000.",
                "solution": "The exact monthly housing expense is $2,000."
            },
            {
                "level": "Level 3: Multi-Category Comparative Analysis",
                "problem": "A company surveys 500 employees regarding commuting methods: 200 drive, 150 take the train, 100 ride the bus, and 50 walk. Compute the percentage and central angle for every category.",
                "step1": "Drive: (200 / 500) × 100% = 40%; Angle = 0.40 × 360° = 144°.",
                "step2": "Train: (150 / 500) × 100% = 30%; Angle = 0.30 × 360° = 108°.",
                "step3": "Bus: (100 / 500) × 100% = 20%; Angle = 0.20 × 360° = 72°.",
                "step4": "Walk: (50 / 500) × 100% = 10%; Angle = 0.10 × 360° = 36°.",
                "solution": "Sum Check: 144° + 108° + 72° + 36° = 360° (Verified Correct)."
            }
        ],
        "pitfalls": [
            "Confusing central angles (sum to 360°) with percentages (sum to 100%). Never label a sector as 90% if its angle is 90°!",
            "Attempting to draw pie charts with negative numbers or non-additive data.",
            "Failing to verify that the sum of all calculated central angles equals exactly 360°.",
            "Misinterpreting relative slice sizes when comparing two different pie charts with unequal total sample sizes."
        ],
        "tips": "Always perform a sum check! Central angles MUST sum to 360° and percentages MUST sum to 100%. Use a protractor aligned at the exact center vertex when constructing circle graphs."
    },
    "Median": {
        "title": "Median & Measures of Central Tendency Master Academic Study Notebook",
        "overview": "The median is a fundamental measure of central tendency that represents the exact middle numerical value of a statistical dataset when arranged in ascending order. Unlike the arithmetic mean, the median is highly resistant to extreme outliers and skewed data distributions.",
        "theory": "In descriptive statistics, the median splits a probability distribution or ordered sample into two equal halves (50% of observations fall below the median and 50% fall above). When dataset size N is odd, the median corresponds to the single central item. When N is even, the median is defined as the arithmetic mean of the two middle observations.",
        "formulas": [
            "Position of Median (Odd N): Position = (N + 1) / 2",
            "Position of Median (Even N): Positions are (N / 2) and (N / 2) + 1",
            "Median Value (Even N): Median = [ Value at (N/2) + Value at (N/2 + 1) ] / 2",
            "Grouped Data Median: L + [ ((N/2) - CF) / f ] × h"
        ],
        "examples": [
            {
                "level": "Level 1: Odd Dataset Median",
                "problem": "Find the median of the following test scores: 85, 92, 67, 74, 88, 95, 71.",
                "step1": "Sort Data in Ascending Order: [67, 71, 74, 85, 88, 92, 95]. Total N = 7 (Odd).",
                "step2": "Find Middle Position: Position = (7 + 1) / 2 = 4th item.",
                "step3": "Identify 4th item in sorted array: 85.",
                "solution": "The median score is 85."
            },
            {
                "level": "Level 2: Even Dataset Median with Outliers",
                "problem": "Find the median of employee salaries ($ in thousands): 45, 50, 42, 120, 48, 52, 46, 55.",
                "step1": "Sort Data in Ascending Order: [42, 45, 46, 48, 50, 52, 55, 120]. Total N = 8 (Even).",
                "step2": "Identify Middle Two Positions: N/2 = 4th item (48) and 5th item (50).",
                "step3": "Compute Average: Median = (48 + 50) / 2 = 98 / 2 = 49.",
                "solution": "The median salary is $49,000 (Notice how the outlier 120 does not distort the median)."
            },
            {
                "level": "Level 3: Frequency Distribution Median",
                "problem": "A survey records student quiz scores out of 5: Score 1 (2 students), Score 2 (5 students), Score 3 (8 students), Score 4 (4 students), Score 5 (1 student). Find the median score.",
                "step1": "Total N = 2 + 5 + 8 + 4 + 1 = 20 students (Even).",
                "step2": "Middle Positions: 10th and 11th students.",
                "step3": "Cumulative Frequencies: Score 1 (2), Score 2 (7), Score 3 (15). Both 10th and 11th items fall in Score 3.",
                "solution": "The median quiz score is 3."
            }
        ],
        "pitfalls": [
            "Calculating the middle element WITHOUT sorting the dataset first! This is the most common exam mistake.",
            "For even N, picking only one of the middle numbers instead of taking their average.",
            "Confusing Median (middle value) with Mode (most frequent value) or Mean (arithmetic average)."
        ],
        "tips": "Always double-check sorting! Cross off elements from both ends (smallest and largest) simultaneously until you reach the center item(s)."
    },
    "Equivalent Fractions": {
        "title": "Equivalent Fractions & Rational Numbers Master Academic Notebook",
        "overview": "Equivalent fractions are fractions that represent the exact same numerical quantity or proportion, even though they possess different numerators and denominators. They are formed by multiplying or dividing both the numerator and denominator by the same non-zero integer.",
        "theory": "The mathematical foundation of equivalent fractions stems from the Multiplicative Identity Property: multiplying any real number by 1 does not alter its value. Since (k / k) = 1 for any integer k ≠ 0, multiplying (a / b) by (k / k) yields (a · k) / (b · k), which is numerically identical to (a / b).",
        "formulas": [
            "Equivalence Rule: (a / b) = (a × k) / (b × k) for k ≠ 0",
            "Simplification Rule: (a / b) = (a ÷ d) / (b ÷ d) where d = GCD(a, b)",
            "Cross-Multiplication Test: (a / b) = (c / d) if and only if (a × d) = (b × c)",
            "Rational Equality: (a / b) - (c / d) = 0"
        ],
        "examples": [
            {
                "level": "Level 1: Building Equivalent Fractions",
                "problem": "Find an equivalent fraction for 3/7 with a denominator of 35.",
                "step1": "Determine the required multiplier for denominator: 35 ÷ 7 = 5.",
                "step2": "Multiply both numerator and denominator by 5: (3 × 5) / (7 × 5).",
                "step3": "Compute result: 15 / 35.",
                "solution": "3/7 is equivalent to 15/35."
            },
            {
                "level": "Level 2: Simplifying to Simplest Terms (GCD)",
                "problem": "Reduce the fraction 48/72 to its simplest irreducible equivalent form.",
                "step1": "Find Greatest Common Divisor GCD(48, 72): Factors of 48 and 72 -> GCD = 24.",
                "step2": "Divide numerator and denominator by 24: (48 ÷ 24) / (72 ÷ 24).",
                "step3": "Compute result: 2 / 3.",
                "solution": "The simplest equivalent fraction is 2/3."
            },
            {
                "level": "Level 3: Cross-Multiplication Equivalence Verification",
                "problem": "Determine whether the fractions 14/21 and 26/39 are equivalent.",
                "step1": "Apply Cross-Product Rule: Check if (14 × 39) == (21 × 26).",
                "step2": "Calculate Left Side: 14 × 39 = 546.",
                "step3": "Calculate Right Side: 21 × 26 = 546.",
                "solution": "Since 546 == 546, the fractions 14/21 and 26/39 are strictly EQUIVALENT."
            }
        ],
        "pitfalls": [
            "Adding or subtracting the same number to numerator and denominator! Example: (3+2)/(4+2) = 5/6 ≠ 3/4.",
            "Multiplying the numerator by one factor and denominator by a different factor.",
            "Forgetting to check if the fraction can be simplified further using GCD."
        ],
        "tips": "Use the cross-multiplication test (a·d = b·c) for instant equivalence verification on exams!"
    },
    "Finding Percents": {
        "title": "Finding Percents & Ratio Proportions Master Academic Notebook",
        "overview": "Percentage represents a ratio, fraction, or dimensional quantity expressed as a fraction of 100 (from Latin 'per centum' meaning 'by the hundred'). Percent calculations are fundamental across financial analysis, interest rates, statistical error margins, and growth rate modeling.",
        "theory": "Mathematically, P% is defined as P / 100. Any percentage calculation involves three interconnected variables: Part (Amount), Whole (Base Total), and Percent (Rate). The fundamental governing equation is Part = (Percent / 100) × Whole.",
        "formulas": [
            "Part = (Percent / 100) × Whole",
            "Percent (%) = (Part / Whole) × 100%",
            "Whole = Part / (Percent / 100)",
            "Percentage Change (%) = [ (New Value - Original Value) / Original Value ] × 100%",
            "Compound Amount = Principal × (1 + r/100)^t"
        ],
        "examples": [
            {
                "level": "Level 1: Finding Percentage of a Number",
                "problem": "Calculate 18% of $450.",
                "step1": "Convert percentage to decimal or fraction: 18% = 18 / 100 = 0.18.",
                "step2": "Multiply by whole amount: 0.18 × 450.",
                "step3": "Compute: 0.18 × 450 = 81.",
                "solution": "18% of $450 is $81."
            },
            {
                "level": "Level 2: Determining Percentage Increase",
                "problem": "An engineering textbook price increased from $80 to $100. Calculate the percentage increase.",
                "step1": "Find absolute change: New Value - Original Value = $100 - $80 = $20.",
                "step2": "Divide by Original Value: $20 / $80 = 0.25.",
                "step3": "Multiply by 100%: 0.25 × 100% = 25%.",
                "solution": "The textbook price increased by 25%."
            },
            {
                "level": "Level 3: Reverse Percentage Calculation",
                "problem": "After receiving a 15% discount, a student bought a laptop for $680. What was the original full price before discount?",
                "step1": "Recognize discounted price percentage: 100% - 15% = 85% of Original Price.",
                "step2": "Set up equation: 0.85 × Original Price = $680.",
                "step3": "Solve for Original Price: $680 / 0.85 = $800.",
                "solution": "The original price of the laptop was $800."
            }
        ],
        "pitfalls": [
            "Dividing by the new price instead of the ORIGINAL price when calculating percentage increase or decrease!",
            "Confusing percentage points with percentage change.",
            "Forgetting to convert percent to decimal (dividing by 100) before multiplication."
        ],
        "tips": "Mental Math Trick: X% of Y is always equal to Y% of X! Example: 16% of 50 = 50% of 16 = 8."
    },
    "Proportion": {
        "title": "Proportions & Ratio Analysis Master Academic Notebook",
        "overview": "A proportion is a mathematical statement asserting that two ratios or rates are strictly equal. Written in the form a/b = c/d or a:b = c:d, proportions govern scale factor conversions, direct/inverse variation, geometric similarity, and engineering unit scaling.",
        "theory": "In the proportion a/b = c/d, the terms a and d are called the 'extremes', while b and c are called the 'means'. The Fundamental Theorem of Proportions states that the product of the extremes equals the product of the means: a · d = b · c. Direct proportion implies y = kx, whereas inverse proportion implies y = k / x.",
        "formulas": [
            "Proportion Equality: (a / b) = (c / d) where b ≠ 0, d ≠ 0",
            "Cross-Product Property (Means-Extremes): a × d = b × c",
            "Direct Variation: y = k × x  (k = Constant of Proportionality)",
            "Inverse Variation: y = k / x  (k = x × y)",
            "Scale Factor: Scale = Model Length / Actual Length"
        ],
        "examples": [
            {
                "level": "Level 1: Solving Unknown Variable in Proportion",
                "problem": "Solve for x in the proportion: 5 / 8 = x / 48.",
                "step1": "Apply Cross-Product Property: 5 × 48 = 8 × x.",
                "step2": "Compute left product: 240 = 8x.",
                "step3": "Divide by 8: x = 240 / 8 = 30.",
                "solution": "x = 30."
            },
            {
                "level": "Level 2: Engineering Scale Model Conversion",
                "problem": "An architectural blueprint uses a scale where 0.5 inches represents 4 feet. If a room measures 3.5 inches on the blueprint, find its actual length in feet.",
                "step1": "Set up proportion: (0.5 in / 4 ft) = (3.5 in / X ft).",
                "step2": "Cross-multiply: 0.5 × X = 4 × 3.5.",
                "step3": "Compute: 0.5X = 14 -> X = 14 / 0.5 = 28 feet.",
                "solution": "The actual room length is 28 feet."
            },
            {
                "level": "Level 3: Inverse Proportion Worker Problem",
                "problem": "If 6 software engineers complete a module in 10 days, how many days will it take 15 engineers working at the same pace?",
                "step1": "Recognize Inverse Proportion: More workers = Fewer days. Formula: Workers₁ × Days₁ = Workers₂ × Days₂.",
                "step2": "Substitute values: 6 × 10 = 15 × D₂.",
                "step3": "Compute: 60 = 15 · D₂ -> D₂ = 60 / 15 = 4 days.",
                "solution": "15 engineers will complete the module in 4 days."
            }
        ],
        "pitfalls": [
            "Cross-multiplying when fractions are being ADDED or MULTIPLIED instead of EQUATED!",
            "Treating inverse variation problems as direct proportions.",
            "Failing to maintain consistent units across corresponding numerators and denominators."
        ],
        "tips": "Always align units across numerators and denominators! (Units A / Units B = Units A / Units B)."
    },
    "Reading a Ruler": {
        "title": "Reading a Ruler & Metric Scale Measurement Master Notebook",
        "overview": "Accurate linear measurement using metric scale rulers is a fundamental skill across physics, mechanical drafting, structural engineering, and manufacturing quality control. Metric rulers utilize a base-10 decimal system divided into meters, centimeters, and millimeters.",
        "theory": "The International System of Units (SI) defines linear distance using the meter (m). Metric rulers display main graduation marks representing centimeters (cm) and minor subdivisions representing millimeters (mm). Each centimeter contains exactly 10 millimeters, meaning 1 mm = 0.1 cm = 0.001 m.",
        "formulas": [
            "Metric Conversion: 1 Meter (m) = 100 Centimeters (cm) = 1,000 Millimeters (mm)",
            "Centimeter to Millimeter: Value (mm) = Value (cm) × 10",
            "Millimeter to Centimeter: Value (cm) = Value (mm) / 10",
            "Measurement Reading = Major Division (cm) + (Minor Marks × 0.1 cm)",
            "Measurement Tolerance / Uncertainty = ± 0.5 × Smallest Scale Division"
        ],
        "examples": [
            {
                "level": "Level 1: Basic Reading Conversion",
                "problem": "An object measure on a metric ruler spans 7 major centimeter marks plus 4 minor millimeter tick marks past 7 cm. Express its length in cm and mm.",
                "step1": "Identify major mark: 7 cm.",
                "step2": "Add minor tick marks: 4 mm = 0.4 cm.",
                "step3": "Total length in cm: 7 + 0.4 = 7.4 cm.",
                "step4": "Convert to mm: 7.4 × 10 = 74 mm.",
                "solution": "Length is 7.4 cm or 74 mm."
            },
            {
                "level": "Level 2: Offset Zero Line Measurement",
                "problem": "An engineer measures a component with a damaged ruler edge. The left edge is aligned at 2.3 cm and the right edge lands at 11.8 cm. Find the exact length.",
                "step1": "Apply Offset Reading Rule: True Length = End Reading - Start Reading.",
                "step2": "Substitute values: Length = 11.8 cm - 2.3 cm.",
                "step3": "Compute: 9.5 cm.",
                "solution": "The actual component length is 9.5 cm (95 mm)."
            },
            {
                "level": "Level 3: Scale Unit Precision & Uncertainty",
                "problem": "A precision rod is measured as 42 mm. If the standard measurement uncertainty of the scale is ± 0.5 mm, calculate the percentage uncertainty.",
                "step1": "Absolute Uncertainty = ± 0.5 mm.",
                "step2": "Percentage Uncertainty = (Absolute Uncertainty / Measured Value) × 100%.",
                "step3": "Compute: (0.5 / 42) × 100% = 0.0119 × 100% = 1.19%.",
                "solution": "Percentage uncertainty is ± 1.19%."
            }
        ],
        "pitfalls": [
            "Aligning the edge of the object with the physical end of the ruler rather than the ZERO calibration line!",
            "Miscounting minor tick marks (confusing 0.5 cm mid-length ticks with 1 cm marks).",
            "Forgetting that 1 cm = 10 mm (not 100 mm)."
        ],
        "tips": "Look straight down at the graduation marks perpendicular to the ruler surface to eliminate parallax measurement errors!"
    },
    "Quadratic Formula": {
        "title": "Quadratic Formula & Polynomial Roots Master Academic Notebook",
        "overview": "The Quadratic Formula provides a universal closed-form algebraic solution for finding the roots (zeros or x-intercepts) of any second-degree polynomial equation in standard form ax² + bx + c = 0, where a ≠ 0.",
        "theory": "Derived by completing the square on the general quadratic equation ax² + bx + c = 0, the Quadratic Formula is x = (-b ± √(b² - 4ac)) / (2a). The expression under the radical Δ = b² - 4ac is called the Discriminant. It dictates the nature of the roots: Δ > 0 yields two distinct real roots; Δ = 0 yields one repeated real root; Δ < 0 yields two complex conjugate roots.",
        "formulas": [
            "Standard Form: a x² + b x + c = 0  (a ≠ 0)",
            "Quadratic Formula: x = [ -b ± √( b² - 4 a c ) ] / ( 2 a )",
            "Discriminant (Δ): Δ = b² - 4 a c",
            "Sum of Roots (Vieta's Formula): x₁ + x₂ = -b / a",
            "Product of Roots (Vieta's Formula): x₁ × x₂ = c / a",
            "Vertex Coordinates (h, k): h = -b / (2a), k = c - (b² / 4a)"
        ],
        "examples": [
            {
                "level": "Level 1: Two Distinct Real Roots (Δ > 0)",
                "problem": "Solve for x in the quadratic equation: x² - 5x + 6 = 0.",
                "step1": "Identify Coefficients: a = 1, b = -5, c = 6.",
                "step2": "Compute Discriminant Δ: b² - 4ac = (-5)² - 4(1)(6) = 25 - 24 = 1 (Δ > 0).",
                "step3": "Apply Quadratic Formula: x = [ -(-5) ± √1 ] / (2 × 1) = [ 5 ± 1 ] / 2.",
                "step4": "Root 1: x = (5 + 1) / 2 = 6 / 2 = 3. Root 2: x = (5 - 1) / 2 = 4 / 2 = 2.",
                "solution": "The real roots are x = 3 and x = 2."
            },
            {
                "level": "Level 2: Repeated Real Root (Δ = 0)",
                "problem": "Solve for x: 4x² - 12x + 9 = 0.",
                "step1": "Coefficients: a = 4, b = -12, c = 9.",
                "step2": "Compute Discriminant: Δ = (-12)² - 4(4)(9) = 144 - 144 = 0.",
                "step3": "Apply Formula: x = [ -(-12) ± √0 ] / (2 × 4) = 12 / 8 = 3 / 2 = 1.5.",
                "solution": "The single repeated real root is x = 1.5."
            },
            {
                "level": "Level 3: Complex Conjugate Roots (Δ < 0)",
                "problem": "Solve for x: x² - 4x + 13 = 0.",
                "step1": "Coefficients: a = 1, b = -4, c = 13.",
                "step2": "Compute Discriminant: Δ = (-4)² - 4(1)(13) = 16 - 52 = -36 (Δ < 0).",
                "step3": "Apply Imaginary Unit i = √(-1): √(-36) = 6i.",
                "step4": "Compute Roots: x = [ 4 ± 6i ] / 2 = 2 ± 3i.",
                "solution": "The complex conjugate roots are x = 2 + 3i and x = 2 - 3i."
            }
        ],
        "pitfalls": [
            "Forgetting to rearrange equations into standard form ax² + bx + c = 0 before identifying a, b, c!",
            "Sign errors when b is negative! Remember -b becomes positive: -(-5) = +5.",
            "Dividing only the radical term by 2a instead of dividing the entire numerator (-b ± √Δ)."
        ],
        "tips": "Always check roots using Vieta's formulas: Sum of roots MUST equal -b/a, and product of roots MUST equal c/a!"
    },
    "Systems of Linear Equations": {
        "title": "Systems of Linear Equations & Matrix Algebra Master Notebook",
        "overview": "A System of Linear Equations consists of two or more linear equations sharing the same set of unknown variables. Solving the system involves finding variable values that simultaneously satisfy all equations in the system.",
        "theory": "Geometrically, each linear equation in two variables represents a straight line on a Cartesian plane. The solution set corresponds to the intersection point(s) of these lines: a unique single solution indicates intersecting lines; infinite solutions indicate identical coincident lines; zero solutions indicate parallel lines. Systems are solved algebraically via Substitution, Elimination, or Cramer's Rule.",
        "formulas": [
            "Standard 2-Variable System: a₁ x + b₁ y = c₁  and  a₂ x + b₂ y = c₂",
            "Cramer's Rule Determinant (D): D = (a₁ b₂ - a₂ b₁)",
            "Cramer's Rule X Solution: x = (c₁ b₂ - c₂ b₁) / D  (D ≠ 0)",
            "Cramer's Rule Y Solution: y = (a₁ c₂ - a₂ c₁) / D  (D ≠ 0)",
            "Slope Comparison: m₁ = -a₁/b₁, m₂ = -a₂/b₂"
        ],
        "examples": [
            {
                "level": "Level 1: Algebraic Elimination Method",
                "problem": "Solve the system of equations:\n1) 2x + 3y = 13\n2) 4x - 3y = 11",
                "step1": "Add equations 1 and 2 to eliminate y: (2x + 4x) + (3y - 3y) = 13 + 11.",
                "step2": "Simplify: 6x = 24 -> x = 4.",
                "step3": "Substitute x = 4 into equation 1: 2(4) + 3y = 13 -> 8 + 3y = 13 -> 3y = 5 -> y = 5/3.",
                "solution": "Solution point: (x = 4, y = 5/3)."
            },
            {
                "level": "Level 2: Algebraic Substitution Method",
                "problem": "Solve the system:\n1) x - 2y = 3\n2) 3x + y = 16",
                "step1": "Isolate x in equation 1: x = 2y + 3.",
                "step2": "Substitute into equation 2: 3(2y + 3) + y = 16.",
                "step3": "Expand and simplify: 6y + 9 + y = 16 -> 7y = 7 -> y = 1.",
                "step4": "Find x: x = 2(1) + 3 = 5.",
                "solution": "Solution point: (x = 5, y = 1)."
            },
            {
                "level": "Level 3: Cramer's Rule Matrix Determinant",
                "problem": "Solve using Cramer's Rule:\n1) 3x + 2y = 7\n2) 5x + 4y = 13",
                "step1": "Main Determinant D = (3)(4) - (5)(2) = 12 - 10 = 2.",
                "step2": "Determinant Dx = (7)(4) - (13)(2) = 28 - 26 = 2 -> x = Dx / D = 2 / 2 = 1.",
                "step3": "Determinant Dy = (3)(13) - (5)(7) = 39 - 35 = 4 -> y = Dy / D = 4 / 2 = 2.",
                "solution": "Solution point: (x = 1, y = 2)."
            }
        ],
        "pitfalls": [
            "Sign distribution errors when subtracting whole equations during elimination!",
            "Confusing parallel lines (D = 0, No Solution) with dependent identical lines (D = 0, Infinite Solutions).",
            "Failing to substitute calculated values back into BOTH original equations for verification."
        ],
        "tips": "Always perform a plug-in check! Substitute (x, y) back into BOTH original equations to confirm 100% accuracy."
    },
    "Multiplication Whole Numbers": {
        "title": "Multiplication of Whole Numbers & Arithmetic Properties Master Notebook",
        "overview": "Multiplication of whole numbers is a fundamental binary operation representing repeated addition. The product of multiplying a multiplier by a multiplicand yields the total combined quantity across equal-sized groups.",
        "theory": "Multiplication is grounded in Peano axioms and set operations. It obeys fundamental algebraic field properties: Commutative Property (a · b = b · a), Associative Property ((a · b) · c = a · (b · c)), and Distributive Property over addition (a · (b + c) = a·b + a·c). Multi-digit multiplication utilizes place-value expansion.",
        "formulas": [
            "Repeated Addition: a × b = b + b + ... + b  (a times)",
            "Distributive Property: a × (b + c) = (a × b) + (a × c)",
            "Commutative Law: a × b = b × a",
            "Associative Law: (a × b) × c = a × (b × c)",
            "Multiplicative Identity: a × 1 = a, Multiplicative Zero: a × 0 = 0"
        ],
        "examples": [
            {
                "level": "Level 1: Mental Math via Distributive Property",
                "problem": "Calculate 14 × 25 using the distributive property.",
                "step1": "Decompose 14 into (10 + 4).",
                "step2": "Apply Distributive Law: (10 + 4) × 25 = (10 × 25) + (4 × 25).",
                "step3": "Compute partial products: 250 + 100 = 350.",
                "solution": "14 × 25 = 350."
            },
            {
                "level": "Level 2: Multi-Digit Standard Algorithm",
                "problem": "Calculate the product of 348 × 46.",
                "step1": "Multiply 348 by 6 (ones place): 348 × 6 = 2,088.",
                "step2": "Multiply 348 by 40 (tens place): 348 × 40 = 13,920.",
                "step3": "Add partial products: 2,088 + 13,920 = 16,008.",
                "solution": "348 × 46 = 16,008."
            },
            {
                "level": "Level 3: Lattice & Area Model Multiplication",
                "problem": "An industrial printing facility produces 1,250 pages per minute. Compute total pages printed in 8 hours (480 minutes).",
                "step1": "Set up expression: 1,250 × 480.",
                "step2": "Factor out trailing zeros: (125 × 48) × 100.",
                "step3": "Compute 125 × 48: 125 × 40 = 5,000; 125 × 8 = 1,000 -> 5,000 + 1,000 = 6,000.",
                "step4": "Append trailing zeros: 6,000 × 100 = 600,000.",
                "solution": "Total pages printed = 600,000 pages."
            }
        ],
        "pitfalls": [
            "Forgetting to write the placeholder zero when moving to tens or hundreds place lines in partial products!",
            "Carrying numbers incorrectly when performing column additions.",
            "Confusing zero multiplication (a × 0 = 0) with identity multiplication (a × 1 = a)."
        ],
        "tips": "Estimate products first! For 348 × 46, estimate 350 × 50 = 17,500 to catch major magnitude errors immediately."
    },
    "Computation with Real Numbers": {
        "title": "Computation with Real Numbers & Order of Operations Master Notebook",
        "overview": "The Real Number System (ℝ) encompasses all rational numbers (integers, fractions, terminating/repeating decimals) and irrational numbers (such as √2, π, e). Computing with real numbers requires strict adherence to mathematical operator precedence rules.",
        "theory": "Real numbers form a complete ordered field. Operator precedence is governed by the PEMDAS / BODMAS convention: Parentheses/Brackets, Exponents/Orders, Multiplication & Division (evaluated left to right), and Addition & Subtraction (evaluated left to right).",
        "formulas": [
            "PEMDAS / BODMAS Order of Operations Hierarchy",
            "Absolute Value Property: |x| = x if x ≥ 0; |x| = -x if x < 0",
            "Signed Operations: (-) × (-) = (+), (-) × (+) = (-)",
            "Exponent Radical Rule: x^(m/n) = n-th root of (x^m)",
            "Distributive Law: a(b + c) = ab + ac"
        ],
        "examples": [
            {
                "level": "Level 1: PEMDAS Order of Operations",
                "problem": "Evaluate expression: 8 + 2 × (3² - 5).",
                "step1": "Evaluate Exponents inside Parentheses: 3² = 9.",
                "step2": "Evaluate Parentheses: (9 - 5) = 4.",
                "step3": "Perform Multiplication: 2 × 4 = 8.",
                "step4": "Perform Addition: 8 + 8 = 16.",
                "solution": "Evaluated result = 16."
            },
            {
                "level": "Level 2: Operations with Radicals & Negatives",
                "problem": "Simplify: -3 × |-7 + 2| + √144 / (-4).",
                "step1": "Evaluate Absolute Value: |-7 + 2| = |-5| = 5.",
                "step2": "Evaluate Radical: √144 = 12.",
                "step3": "Multiply and Divide: -3 × 5 = -15; 12 / (-4) = -3.",
                "step4": "Combine: -15 + (-3) = -18.",
                "solution": "Simplified result = -18."
            },
            {
                "level": "Level 3: Complex Nested Fraction Expression",
                "problem": "Simplify: [ (-2)³ + 4 × (5 - 8) ] / [ √49 - (3 - 6)² ].",
                "step1": "Numerator: (-2)³ = -8; (5 - 8) = -3; 4 × (-3) = -12 -> Numerator = -8 + (-12) = -20.",
                "step2": "Denominator: √49 = 7; (3 - 6) = -3; (-3)² = 9 -> Denominator = 7 - 9 = -2.",
                "step3": "Final Division: -20 / -2 = +10.",
                "solution": "Evaluated result = +10."
            }
        ],
        "pitfalls": [
            "Performing addition before multiplication! Example: In 5 + 3 × 2, writing 8 × 2 = 16 instead of 5 + 6 = 11.",
            "Square of a negative number error: Note that -3² = -9, whereas (-3)² = +9!",
            "Evaluating multiplication before division when division appears to the left (evaluate left-to-right!)."
        ],
        "tips": "Always evaluate left-to-right for operators of equal rank (Multiplication/Division, Addition/Subtraction)!"
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
    "Mode": {
        "embed_url": "https://www.youtube.com/embed/B1HEzNTGeZ4",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Mode",
        "edu_url": "https://www.youtube.com/results?search_query=Mode+statistics+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Mode+statistics+math+tutorial"
    },
    "Mean": {
        "embed_url": "https://www.youtube.com/embed/B1HEzNTGeZ4",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Mean",
        "edu_url": "https://www.youtube.com/results?search_query=Mean+statistics+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Mean+statistics+math+tutorial"
    },
    "Proportion": {
        "embed_url": "https://www.youtube.com/embed/GO56OiUjzp0",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Proportion",
        "edu_url": "https://www.youtube.com/results?search_query=Proportions+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Proportions+math+tutorial"
    },
    "Percent Of": {
        "embed_url": "https://www.youtube.com/embed/Uf-LCC0v7C4",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Percent+Of",
        "edu_url": "https://www.youtube.com/results?search_query=Finding+percent+of+a+number+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Percent+of+a+number+math+tutorial"
    },
    "Solving Systems of Linear Equations": {
        "embed_url": "https://www.youtube.com/embed/bAerID24QJ0",
        "khan_url": "https://www.khanacademy.org/search?referer=%2F&page_search_query=Systems+of+Linear+Equations",
        "edu_url": "https://www.youtube.com/results?search_query=Solving+systems+of+linear+equations+math+lesson",
        "web_url": "https://www.google.com/search?tbm=vid&q=Systems+of+linear+equations+math+tutorial"
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
    return redirect(url_for('instructions'))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route("/instructions")
@login_required
def instructions():
    student_name = session.get('student_name', 'Student Candidate')
    register_number = session.get('register_number', '210101001')
    department = session.get('department', 'Computer Science')
    return render_template("instructions.html", student_name=student_name, register_number=register_number, department=department)

@app.route("/quiz")
@login_required
def quiz():
    filter_skill = request.args.get('skill')
    all_questions = load_questions(filter_skill=filter_skill)
    
    if filter_skill:
        questions = all_questions
    else:
        # Group questions by skill to ensure balanced 10-question assessment across 10 distinct skills
        skill_groups = {}
        for q in all_questions:
            s = q.get('skill_name', q.get('skill', 'General Skill'))
            if s not in skill_groups:
                skill_groups[s] = []
            skill_groups[s].append(q)
            
        sampled = []
        for s, q_list in skill_groups.items():
            if len(sampled) < 10:
                sampled.append(random.choice(q_list))
                
        # Fill remaining slots if fewer than 10 skills
        if len(sampled) < 10:
            remaining = [q for q in all_questions if q not in sampled]
            needed = 10 - len(sampled)
            if remaining:
                sampled.extend(random.sample(remaining, min(needed, len(remaining))))
            
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

    for q in questions:
        q_id_str = str(q['id'])
        q_id_int = int(q['id'])
        skill = q.get('skill_name', q.get('skill', 'General Skill'))
        raw_correct_ans = str(q.get('answer', '')).strip()
        options = q.get('options', [])

        if skill not in skill_stats:
            skill_stats[skill] = {"total": 0, "correct": 0, "wrong": 0}
        
        skill_stats[skill]["total"] += 1
        
        # Look up user answer STRICTLY by question ID (prevents key swapping)
        user_ans_idx = user_answers.get(q_id_str)
        if user_ans_idx is None:
            user_ans_idx = user_answers.get(q_id_int)

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
        except Exception as dl_err:
            print(f"PyTorch LSTM DKT Prediction Note: {dl_err}")

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
            "skill_names_json": json.dumps([]),
            "skill_scores_json": json.dumps([])
        }
    return render_template("result.html", result_data=result_data)

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
            <p>Knowledge Gap Detection & Personalized Learning Recommendation System &bull; Final Year Engineering Project</p>
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