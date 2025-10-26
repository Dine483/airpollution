# airpollution
Project OverviewThis project performs a preliminary data science analysis to explore the relationship between urban air pollutants, weather variables, and the occurrence of respiratory cases. It uses a synthetic dataset to train and evaluate a Linear Regression model, identifying the most influential environmental factors on public respiratory health.
Goal & SDG AlignmentThe core goal is to provide data-driven insights for public health policy. This project aligns with:SDG Target 3.4 (Good Health and Well-Being): By modeling the risk of non-communicable respiratory diseases.SDG Target 13.1 (Climate Action): By strengthening adaptive capacity to environmental health hazards like air pollution.
Setup Guide (Step-by-Step)Follow these steps to set up the environment and run the analysis.
1. Prerequisites (Software Installation)You must have Python 3 installed. Using Anaconda is highly recommended for managing environments and packages.
Install Anaconda/Miniconda: Download and install from the official website.
Open Anaconda Prompt (Windows) or Terminal (macOS/Linux).
2. Environment SetupCreate an isolated environment for the project and install the necessary libraries.Bash#
3.  1. Create a new environment
>conda create -n airproj python=3.10 -y

# 2. Activate the environment
>conda activate airproj

# 3. Install required libraries
>pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab

3. Folder StructureSet up the clean project directory structure.Bash# Create the main folder
>mkdir air-pollution-project
>cd air-pollution-project

# Create required sub-directories
>mkdir data plots src notebooks

4. Running the Analysis (Two Options)You have two primary ways to run the analysis:Option A: Jupyter Notebook (Recommended for Report/Presentation)The Jupyter Notebook (Untitled.ipynb or analysis.ipynb) allows you to run the code cell-by-cell and view plots and results inline.Place the File: Put the file Untitled.ipynb into the notebooks/ directory.
5. Start Jupyter Lab:Bashconda activate airproj
jupyter lab
Run: A browser will open. Navigate to notebooks/Untitled.ipynb and run all cells sequentially.Option B: Python Script (For Reproducible Runs)The Python script (run_analysis.py) runs the entire pipeline from start to finish, saving all outputs to files.
Place the File: Put the file run_analysis.py into the src/ directory.
Run the script:>Bashconda activate airproj
>python src/run_analysis.py

 Project StructurePathDescriptionair-pollution-project/Root Directory
├─ data/Stores the processed dataset (air_resp_data.csv).
├─ plots/Stores all generated plots (coefficients.png, corr_matrix.png, etc.).
├─ src/Contains the single runnable Python script (run_analysis.py).
└─ notebooks/Contains the step-by-step interactive analysis (Untitled.ipynb).
 Key ResultsThe Linear Regression model provided the following insights from the analysis:Model Performance: $\text{R}^2$ Score of $\approx 0.74$ (High predictive power on the synthetic data).Most Impactful Features (Positive Correlation): $\text{NO}_2$ and $\text{PM}2.5$ showed the highest positive coefficients, indicating they are the strongest predictors for an increase in respiratory cases.Mitigating Factors: $\text{O}_3$ and Humidity showed slight negative coefficients.
 Future EnhancementsFor a transition to a real-world project, consider these improvements:Use Real Data: Replace the synthetic data with actual time-series air quality data (e.g., OpenAQ, CPCB) and hospital admissions/mortality data.Advanced Modeling: Use Poisson Regression (appropriate for count data like hospital visits) and integrate lagged features to capture the delayed effect of pollution.Confounder Control: Include temporal confounders like day-of-week, season, or viral activity (e.g., flu season).
