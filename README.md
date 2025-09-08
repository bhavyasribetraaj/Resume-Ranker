# Resume-Ranker
Resume Ranker — Pro+

Resume Ranker — Pro+ is an advanced AI-powered web application designed to streamline the recruitment process by analyzing and ranking resumes based on their relevance to a job description (JD). Built with Streamlit, scikit-learn, and Plotly, it offers robust features like semantic similarity analysis, keyword matching, candidate clustering, and interactive visualizations.

✨ Features

Multi-Format Resume Parsing: Supports PDF, DOCX, TXT, HTML, CSV, XLSX, and ZIP files containing multiple resumes.
AI-Powered Ranking: Uses TF-IDF or sentence embeddings (via sentence-transformers) to compute similarity between resumes and job descriptions.
Keyword Extraction & Highlighting: Extracts and highlights key job requirements in resumes.
Advanced Analytics: Includes candidate clustering, skills analysis, experience and education extraction, and resume quality scoring.
Interactive Visualizations: Dynamic charts (histograms, bar charts, scatter plots, radar charts) using Plotly for insightful candidate comparisons.
Filtering & Comparison: Filter candidates by score, experience, education, or skills, and compare up to four candidates side-by-side.
Export Capabilities: Export results as CSV, summary reports as TXT, and individual candidate reports as PDF.
Customizable Settings: Adjust semantic weight, enable lemmatization, and toggle advanced features like clustering and skills analysis.

📋 Requirements

Python 3.8+
Libraries:pip install streamlit pandas numpy scikit-learn plotly fpdf PyMuPDF python-docx beautifulsoup4 sentence-transformers nltk



Optional dependencies:

PyMuPDF for PDF parsing
python-docx for DOCX parsing
beautifulsoup4 for HTML parsing
sentence-transformers for semantic embeddings
nltk for lemmatization
fpdf for PDF report generation

🚀 Getting Started

Clone the Repository:
git clone https://github.com/yourusername/resume-ranker-pro.git
cd resume-ranker-pro


Install Dependencies:
pip install -r requirements.txt


Run the Application:
streamlit run app.py


Access the App:Open your browser and navigate to http://localhost:8501.


🖥️ Usage

Upload Resumes:

Upload one or more resume files (PDF, DOCX, TXT, HTML, CSV, XLSX, or ZIP).
ZIP files can contain multiple resumes for batch processing.


Provide Job Description:

Paste the job description or upload a file (PDF, DOCX, TXT, HTML).


Configure Settings:

Choose between TF-IDF or embeddings for similarity computation.
Adjust semantic weight, keyword extraction, and filtering options (e.g., minimum score, experience, education level).
Enable/disable features like clustering, quality scoring, or skills analysis.


Analyze & Explore:

Click "Analyze & Rank Candidates" to process resumes.
View ranked candidates, detailed analytics, and visualizations.
Use filters to narrow down candidates or compare up to four candidates.


Export Results:

Download results as CSV, summary reports as TXT, or individual candidate reports as PDF.



🛠️ How It Works

Text Extraction: Parses resumes and job descriptions into text using format-specific libraries (PyMuPDF, python-docx, etc.).
Text Processing: Cleans and tokenizes text, with optional lemmatization using NLTK.
Similarity Scoring:
Computes semantic similarity using sentence embeddings (if available) or TF-IDF.
Combines semantic and keyword coverage scores with customizable weights.


Feature Extraction:
Extracts experience, education, and technical skills.
Calculates resume quality based on length, contact info, sections, action verbs, and quantified achievements.


Clustering: Groups candidates into tiers (e.g., Top, Mid, Entry) using K-Means clustering.
Visualization: Generates interactive charts for score distribution, candidate rankings, and clustering analysis.

📊 Example Output

Ranked Candidates: Displays candidates with scores, semantic similarity, keyword coverage, experience, and education.
Visualizations:
Score distribution histogram
Top candidates bar chart
Experience vs. score scatter plot
Candidate clustering scatter plot
Skills and education distribution pie charts


Reports: CSV for full results, TXT for summaries, and PDF for individual candidate reports.

🔧 Customization

Modify Scoring: Adjust the weight_semantic slider to prioritize semantic similarity or keyword coverage.
Extend Skills: Update the extract_skills function to include additional skill categories or keywords.
Add Visualizations: Extend Plotly charts in the visualization section for custom analytics.
Enhance Parsing: Add support for new file formats by extending the parse_uploaded_file function.

📝 Notes

Dependencies: Some features (e.g., PDF parsing, embeddings) require optional libraries. Install them for full functionality.
Performance: Processing large numbers of resumes or ZIP files may require significant memory and CPU.
Error Handling: The app gracefully handles missing dependencies by falling back to TF-IDF or skipping unavailable features.
NLTK Data: Ensure punkt and wordnet are downloaded for lemmatization:import nltk
nltk.download('punkt')
nltk.download('wordnet')





Built with Streamlit for the web interface.
Powered by scikit-learn for machine learning and sentence-transformers for embeddings.
Visualizations by Plotly.
Text processing with NLTK and BeautifulSoup.
