- Financial Analysis is a critical means to assess business performance and the consequences of poor analysis can involve costing billions of dollars to facilitate high-quality time and decision-making professionals such as analysts or investors will perform complex quantity analysis to select information from financial reports and such analysis.
  
- The sheer volume of financial statements makes it difficult for humans to access and analyze a business's financials. Robust numerical reasoning likewise faces unique challenges in this domain. In this work, we focus on answering deep questions over financial data, aiming to automate the analysis of a large corpus of financial documents. In contrast to existing tasks on general domain, the finance domain includes complex numerical reasoning and understanding of heterogeneous representations. To facilitate analytical progress, we propose a new large-scale dataset, FinQA, with Question-Answering pairs over Financial reports, written by financial experts.
  
- Here, given Input Data in the form of Structured Tables and Unstructured Text , we need to find out the Solutions as per the Question asked , as per the Mathematical Calculations.
GITHUB LINK: https://github.com/czyssrs/FinQA/tree/main
KAGGLE LINK: https://www.kaggle.com/datasets/visalakshiiyer/question-answering-financial-data
Papers with Code LINK: https://paperswithcode.com/paper/finqa-a-dataset-of-numerical-reasoning-over
HTML Page for an Example: https://finqasite.github.io/explore.html
CodaLab Competition Page: https://codalab.lisn.upsaclay.fr/competitions/1846#learn_the_details

- Data consists of 4 JSON files: Training Data , Validation Data , Test Data and Private Test Data.
Each Row in every Dataset provided consists of a specific passage considered from a Financial Report. And the Data is being considered based on the publicly available earnings reports of S&P 500 companies from 1999 to 2019, collected in the FinTabNet dataset.

- Every Row of the JSON Data is represented as shown below:

"pre_text": the texts before the table (list of strings);
"post_text": the text after the table (list of strings);
"filename": Name of the File from which the Financial Report has been considered. The filename syntax consists of the Stock Ticker Symbols for the publicly traded companies on stock exchanges, the Year of Report and the Page Number from which the Repot has been referred;
"table_ori": The Original Table;
"table": The Table with all text as Lower-Case and Special Characters Removed ;

"qa": {
  "question": The Question,
  "answer": The Answer,
  "explanation": The Explanation to the Answer,
  "ann_table_rows": Annotated Table Rows (denotes the row number(s) using which the answers have been fetched )
  "ann_text_rows": Annotated Text Rows (denotes the row number(s) using which the answers have been fetched )
  "steps": [{"op": "","arg1": "","arg2": "","res": ""}],
  (Steps denote all the Operations performed between any 2 Arguments "arg1","arg2" for any operation "op" giving "res")
  "program": the reasoning program to get the Result,
  "gold_inds": the gold supporting facts,
  "exe_ans": the gold execution result,
  "tfidftopn": the Top-N facts from Text Data after performing TF-IDF,
  "program_re": the reasoning program in nested format,
  "model_input": Sentences which are considered as Input for the Model to get the Result,
}

"id": unique example id. composed by the original report name plus example index for this report. 

"table_retrieved" / "text_retrieved" / "table_retrieved_all" / "text_retrieved_all" : 
These 4 columns contains information about Data Extraction, Processing, or Retrieval Operations, and each column contains entries for Scores and Table Row Number / Text Row Number.
