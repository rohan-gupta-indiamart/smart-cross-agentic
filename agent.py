# gcloud auth login
# gcloud storage cp "gs://hackathon-problem-statements/smart-cross-enrichment-category/Cross-Enrichment Agent Hackathon Data 11th & 12th July.xlsx" ./smart-classifier/input_data.xlsx  # noqa: E501
import traceback
from typing import List, Dict
import os 
from google.adk.agents import Agent
import pandas as pd
#from . import ProductTextAnalysisAgent
#from __init__ import ProductTextAnalysisAgent
from product_agent import ProductTextAnalysisAgent
import torch

def read_sheet_from_xlsx(file_path, sheet_index):
    """
    Reads the content of the sheet at given index from an XLSX file into a pandas DataFrame.


    Args:
        file_path (str): The path to the XLSX file. 
        sheet_index (int): The index of sheet which needs to be read from the XLSX file.


    Returns:
        pandas.DataFrame: A DataFrame containing the data from the given sheet.
                          Returns None if the file is not found or an error occurs. 
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_index)
        print(f"Successfully read the {sheet_index}th sheet from: {file_path}")
        df=df.ffill()


        if df is not None:
            print("\nFirst 10 rows of the DataFrame:")
            print(df.head(10))


            print("\nColumn names:")
            print(df.columns)


            print("\nDataFrame Info:")
            df.info()


        return df
    except FileNotFoundError: 
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e: 
        print(f"An error occurred while reading the Excel file: {e}")
        return None


# file_name = './smart-classifier/input_data.xlsx' # Replace with the actual path to your file
# Get the absolute path to the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the Excel file
file_name = os.path.join(script_dir, 'input_data.xlsx') 

target_vs_related_mcat_df = read_sheet_from_xlsx(file_name,0).head(8)
"""
target_vs_related_mcat_df is DataFrame containing below columns:
 0   Target MCAT ID   float64
 1   Target MCAT      object
 2   Related MCAT ID  int64  
 3   Related MCAT     object
 4   Dataset          float64
"""

index=0
targetMcatID=int(target_vs_related_mcat_df['Target MCAT ID'][index])
targetMcatName=target_vs_related_mcat_df['Target MCAT'][index]
datasetNumber=int(target_vs_related_mcat_df['Dataset'][index])
setSheetName = 'Set '+str(datasetNumber)+' Seller Products'

print(targetMcatID,targetMcatName,datasetNumber,setSheetName)

set_seller_products_df = read_sheet_from_xlsx(file_name,setSheetName)#3
"""
set_seller_products_df is DataFrame containing below columns:
 0   glusr_usr_id         int64 
 1   product id           int64 
 2   product name         string
 3   product description  html text
 4   primary image        url
 5   specifications       map
"""
print("data loaded successfully. starting to process now")
product_details_df = set_seller_products_df[['glusr_usr_id','product id', 'product name', 'product description','specifications']]

print("processing for "+targetMcatName,product_details_df.head())

def get_related_mcat_ids(target_mcat_id: int) -> list[int]: 
    """
    Retrieves a list of all 'Related MCAT ID's for a given 'Target MCAT ID'.


    Args:
        target_mcat_id (int): The Target MCAT ID to search for. 


    Returns:
        list: A list of 'Related MCAT ID's associated with the given
              Target MCAT ID. Returns an empty list if the Target MCAT ID
              is not found.
    """
    df=target_vs_related_mcat_df
    # Ensure the column names match exactly, including leading/trailing spaces if any
    # Based on your df.head() output, there are leading/trailing spaces in column names.
    # It's good practice to strip them for robust indexing.
    df.columns = df.columns.str.strip()


    # Filter the DataFrame to get rows where 'Target MCAT ID' matches the input
    filtered_df = df[df['Target MCAT ID'] == target_mcat_id] 


    # Extract the 'Related MCAT ID' column from the filtered DataFrame
    # and convert it to a list.
    # .dropna() is used in case there are any NaN values in the 'Related MCAT ID' column
    # that you don't want in your list.
    related_ids = filtered_df['Related MCAT ID'].dropna().tolist()


    if len(related_ids) == 0:
        filtered_df = df[df['Related MCAT ID'] == target_mcat_id]
        related_ids = filtered_df['Target MCAT ID'].dropna().tolist()
        related_ids = [int(x) for x in related_ids]


    return related_ids

def analyze_product(product_data: pd.DataFrame) -> Dict:
    """
    Analyzes a single product and returns a dictionary of results. 
    This is a placeholder function to represent the functionality of a 
    ProductTextAnalysisAgent. You should replace this with your actual 
    implementation. 

    Args:
        product_data (pd.DataFrame): Pandas DataFrame containing product information. 
            Needs columns like 'Product ID', 'Product Name', and potentially others. 

    Returns:
        Dict: A dictionary containing the analysis results for the product, including:
            'product_id', 'product_name', and a placeholder 'analysis_result'.
    """
    try:
        product_id = product_data['product id'].iloc[0]
        product_name = product_data['product name'].iloc[0]

        # Placeholder analysis - replace with actual logic
        analysis_result = "Product analysis pending implementation." 

        return {
            "product_id": product_id,
            "product_name": product_name,
            "analysis_result": analysis_result,
        }
    except KeyError as e:
        print(f"Missing column in product data: {e}")
        return {}


def get_product_details_and_targer_mcat_name(product_id: int) -> dict:
    """
    Retrieves the 'glusr_usr_id', 'product id', 'product name', 'product description', 'specifications' and 'target_mcat_name' for a given 'product id'.

    Args:
        product_id (int): The product_id to search for.

    Returns:
        dict: A dict with keys 'glusr_usr_id', 'product id', 'product name', 'product description', 'specifications' and 'target_mcat_name'. Returns None if the product_id is not found.
    """
    df=product_details_df
    # Ensure the column names match exactly, including leading/trailing spaces if any
    # Based on your df.head() output, there are leading/trailing spaces in column names.
    # It's good practice to strip them for robust indexing.
    df.columns = df.columns.str.strip()

    # Filter the DataFrame to get rows where '' matches the input
    filtered_df = df[df['product id'] == product_id]
    if not filtered_df.empty:
        product_details = filtered_df.iloc[0].to_dict()
        product_details['target_mcat_name'] = targetMcatName
        return product_details
    return None

    # Extract the 'Related MCAT ID' column from the filtered DataFrame
    # and convert it to a list.
    # .dropna() is used in case there are any NaN values in the 'Related MCAT ID' column
    # that you don't want in your list.
    #related_ids = filtered_df['Related MCAT ID'].dropna().tolist()

    # if len(related_ids) == 0:
    #     filtered_df = df[df['Related MCAT ID'] == target_mcat_id]
    #     related_ids = filtered_df['Target MCAT ID'].dropna().tolist()
    #     related_ids = [int(x) for x in related_ids]

    # return related_ids

print(get_related_mcat_ids(658771))
print(get_related_mcat_ids(9211))
print(get_product_details_and_targer_mcat_name(283188239))
print(get_product_details_and_targer_mcat_name(280896279))

# Initialize the agent
product_analysis_agent = ProductTextAnalysisAgent(target_vs_related_mcat_df, device="cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

root_agent = Agent(
    name="smart_cross_enrichment_category",
    model="gemini-2.0-flash",
    description=(
        "Agent to orchestrate product analysis, combining results from multiple specialized agents."
    ),
    instruction=(
        "You are a helpful agent that analyzes product information to determine its relevance" 
        " to different categories (MCATs). You call upon specialized agents for different" 
        " aspects of the analysis and combine their results into a final verdict."    ), 
    tools=[get_related_mcat_ids, get_product_details_and_targer_mcat_name, product_analysis_agent.analyze_product],
)


# --- Test Case Implementation --- 

# Load data from Excel file
#file_name = './smart-classifier/input_data.xlsx'
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(script_dir, 'input_data.xlsx') 
try:
    target_related_mcats_df = read_sheet_from_xlsx(file_name, 0)  # "Target vs Related Mcats"
    set1_products_df = read_sheet_from_xlsx(file_name, 3)  # "Set 1 Seller Products"

    if target_related_mcats_df is None or set1_products_df is None:
        raise ValueError("Failed to load data from Excel file.")

    # Clean column names in target_related_mcats_df
    target_related_mcats_df.columns = target_related_mcats_df.columns.str.strip() 

    # Assuming 'Related MCATs' column is a string representation of a list, convert it
    # You might need to adjust the column name if it's different in your sheet
    if "Related MCAT" in target_related_mcats_df.columns:
        target_related_mcats_df["Related MCATs"] = (
            target_related_mcats_df["Related MCAT"]
            #target_related_mcats_df["Related MCAT"].apply(
            #    lambda x: eval(x) if isinstance(x, str) else x
            #)
        )
    else:
        raise KeyError("'Related MCATs' column not found in 'Target vs Related Mcats' sheet.")

    # Initialize the ProductTextAnalysisAgent
    #product_analysis_agent = ProductTextAnalysisAgent(
    #    target_related_mcats_df, device="cpu"  # Or "cuda" if you have a GPU and CUDA set up
    #)

    # Process each product in Set 1
    results = []
    for index, product_row in set1_products_df.iterrows():
        # Ensure product_row has the necessary columns
        if all( 
            col in product_row.index
            for col in [
                "product id",
                "product name",
                "product description",
                "specifications",
                #"MCAT ID where the product is currently mapped",
                "primary image",  # Assuming this column exists
            ]
        ):
            #product_data = pd.DataFrame([product_row])  # Create a single-row DataFrame
            product_data = get_product_details_and_targer_mcat_name(product_row['product id'])  # Create a single-row DataFrame
            analysis_result = product_analysis_agent.analyze_product(product_data)
            if analysis_result:  # Only append if analysis was successful
                results.append(analysis_result)
            break               
        else:
            print(f"Skipping product due to missing columns: {product_row['Product ID']}")

    # Create a DataFrame from the results and print it
    if results:
        results_df = pd.DataFrame(results)
        print("\n--- Product Analysis Results ---")
        print(results_df.to_markdown(index=False))
    else:
        print("\nNo products were successfully analyzed.")

except Exception as e:
    print(f"An error occurred during test case execution: {e}")
    traceback.print_exc()
