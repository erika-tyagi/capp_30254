## Assignment 5: Machine Learning Pipeline III

The goal of this homework assignment is to fix and update your ML pipeline. 

The problem is to predict if a project on donorschoose will not get fully funded within 60 days of posting. This prediction is being done at the time of posting so you can only use data available to you at that time. The data is a file that has one row for each project posted with a column for date_posted (the date the project was posted) and a column for date_fully_funded (the date the project was fully funded - assumption for this assignment is that all projects were fully funded eventually). The problem and data are adopted from here: https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose/data. 

The goal of this assignment is to improve the pipeline by adding more classifiers, experimenting with different parameters for these classifiers, adding additional evaluation metrics, and creating a temporal validation function to create training and test sets over time. 

The raw data (in CSV format) is stored in the data folder, and the three components of the assignment are contained in the following files: 

- Coding: pipeline_v3.py
- Analysis: 
	- homework_5_analysis.ipynb 
	- evaluation_results.csv (full table with evaluation metrics across temporal splits, classifiers, parameters, k-percentage thresholds) 
- Report: homework_5_report.ipynb 