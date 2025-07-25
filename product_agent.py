import os
from typing import List, Dict, Tuple

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch


class ProductTextAnalysisAgent:
    """
    A class to analyze product text and classify its relevance to a target MCAT.
    """

    def __init__(self, target_mcat_data: pd.DataFrame, device: str = "cpu"):
        """
        Initializes the ProductTextAnalysisAgent.

        Args:
            target_mcat_data (pd.DataFrame): DataFrame containing target MCAT and related MCATs.
                Should have columns like "Target MCAT", "Related MCATs".
            device (str):  The device to use for computations, either "cpu" or "cuda".  Defaults to "cpu".
        """
        self.target_mcat_data = target_mcat_data
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # You can change the model here
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(self.device) # And here

    def _get_target_and_related_mcats(self, product_data: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Retrieves the target MCAT and related MCATs based on the product's current mapping.

        Args:
            product_data (pd.DataFrame): DataFrame containing product information.
              Needs a 'MCAT ID where the product is currently mapped' column

        Returns:
            Tuple[str, List[str]]: A tuple containing the target MCAT and a list of related MCATs.
        Raises:
            ValueError: If the product's current MCAT is not found in the target MCAT data.
        """
        """
        current_mcat = product_data['MCAT ID where the product is currently mapped'].iloc[0]  # Assuming one product at a time
        target_mcat_row = self.target_mcat_data[
            self.target_mcat_data['Related MCATs'].apply(lambda x: current_mcat in x)
        ]
        if target_mcat_row.empty:
            raise ValueError(f"Current MCAT '{current_mcat}' not found in target MCAT data.")

        target_mcat = target_mcat_row["Target MCAT"].iloc[0]
        related_mcats = target_mcat_row["Related MCATs"].iloc[0]
        return target_mcat, related_mcats
        """
        return "Shampoo Third Party Manufacturing", []


    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Generates embeddings for a list of texts using the pre-trained model.

        Args:
            texts (List[str]): A list of strings to embed.

        Returns:
            torch.Tensor: A tensor containing the embeddings for each text.
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
        if "Rekosa Onion Shampoo With Conditioner" in texts[0]:
            print("Rekosa Onion Shampoo With Conditioner found in texts. printing inputs:")
            print(inputs)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if "Rekosa Onion Shampoo With Conditioner" in texts[0]:
            print("Rekosa Onion Shampoo With Conditioner found in texts. printing outputs mean:")
            print(outputs.last_hidden_state.mean(dim=1))
        return outputs.last_hidden_state.mean(dim=1)

    def classify_relevance(self, product_name: str, description: str, isqs: str, target_mcat: str) -> float:
        """
        Classifies the relevance of a product to a target MCAT.

        Args:
            product_name (str): The name of the product.
            description (str): The product description.
            isqs (str): Industry-specific questions related to the product.
            target_mcat (str): The target MCAT for relevance classification.

        Returns:
            float: A confidence score (between 0 and 1) representing the product's relevance to the target MCAT.
        """
        product_text = f"{product_name} {description} {isqs}"
        embeddings = self._get_embeddings([product_text, target_mcat])
        similarity_score = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
        return max(0, min(1, (similarity_score + 1) / 2))  # Scale to [0, 1]


    def extract_keywords(self, product_name: str, description: str, isqs: str, related_mcats: List[str]) -> List[str]:
        """
        Extracts latent keywords from product information, aligning with related MCATs.  This is a simplified version and
        could be improved with more advanced techniques (e.g. topic modeling, TF-IDF).

        Args:
            product_name (str): The name of the product.
            description (str): The product description.
            isqs (str): Industry-specific questions related to the product.
            related_mcats (List[str]): A list of related MCATs.

        Returns:
            List[str]: A list of keywords extracted from the product information.  Currently returns the most
             relevant related MCAT as a placeholder.
        """
        product_text = f"{product_name} {description} {isqs}"
        embeddings = self._get_embeddings([product_text] + related_mcats)
        similarities = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1:])
        most_similar_index = similarities.argmax()
        return [related_mcats[most_similar_index]]


    #def analyze_product(self, product_data: pd.DataFrame) -> Dict:
    def analyze_product(self, product_data: Dict) -> Dict:
        """
        Analyzes a single product and returns a dictionary of results. It takes all product details along with some target mcat and checks how much is the target mcat relevant to the product on basis of the text.

        Args:
            product_data (Dict):  Dict containing product information. Requires columns
              'product id', 'product name', 'product description', 'specifications' and 'target_mcat_name' and expects single row dataframe as input.

        Returns:
            Dict: A dictionary containing the analysis results for the product, including:
              'product_id', 'product_name', 'remarks', and 'final_conclusion'.
        """
        try:
            #target_mcat, related_mcats = self._get_target_and_related_mcats(product_data)
            target_mcat = product_data['target_mcat_name']
            product_id = product_data['product id']#.iloc[0]
            product_name = product_data['product name']#.iloc[0]
            #primary_image = product_data['primary image'].iloc[0]
            description = product_data['product description']#.iloc[0]
            isqs = product_data['specifications']#.iloc[0]
            #mcat=prd['MCAT ID where the product is currently mapped']

            relevance_score = self.classify_relevance(product_name, description, isqs, target_mcat)
            #keywords = self.extract_keywords(product_name, description, isqs, related_mcats)

            # Basic remarks and conclusion generation.  Improve based on your specific needs.
            remarks = f"Keywords related to: {product_name}"#{', '.join(keywords)}"
            if relevance_score > 0.5:
                conclusion = f"Product may be relevant to {target_mcat} (Confidence: {relevance_score:.2f})"
            else:
                conclusion = f"Product appears less relevant to {target_mcat} (Confidence: {relevance_score:.2f})"

            return {
                "product_id": product_id,
                "product_name": product_name,
                #"primary_image": primary_image,
                "Remarks": remarks,
                "Final Conclusion": conclusion,
            }
        except KeyError as e:
            print(f"Missing column in product data: {e}")
            return {}
        except ValueError as e:
            print(f"Error processing product: {e}")
            return {}


# Example Usage:
#  Assuming you have 'set1_products.csv' and 'target_related_mcats.csv'

# Load data - Replace with your actual file paths and format if different (e.g., Excel, database)
try:
    #set1_products_df = pd.read_csv("set1_products.csv")  # Contains Product ID, Product Name, primary image, etc.
    #target_related_mcats_df = pd.read_csv("target_related_mcats.csv")  # Contains 'Target MCAT', 'Related MCATs'
    # Read the Excel file
    #input_excel = "input_data.xlsx"  # Make sure this file exists in the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_excel = os.path.join(script_dir, 'input_data.xlsx')
    # Read the relevant sheets by name or index
    set1_products_df = pd.read_excel(input_excel, sheet_name="Set 1 Seller Products")
    target_related_mcats_df = pd.read_excel(input_excel, sheet_name="Target vs Related MCATs")
    # Assuming 'Related MCATs' column is string representation of a list, convert it to a list.
    #target_related_mcats_df["Related MCATs"] = target_related_mcats_df["Related MCAT"].apply(eval)
    #target_related_mcats_df["Related MCATs"] = target_related_mcats_df["Related MCAT"].astype(str).str.split(',')
    target_related_mcats_df["Related MCATs"] = target_related_mcats_df["Related MCAT"] # Assuming it's already a list or similar structure


except FileNotFoundError:
    print("Ensure 'set1_products.csv' and 'target_related_mcats.csv' exist in the current directory.")
    exit()
except Exception as e:
    print(f"Error loading data files: {e}")
    exit()


"""
# Initialize the agent
agent = ProductTextAnalysisAgent(target_related_mcats_df, device="cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Process each product in Set 1
results = []
for index, product_row in set1_products_df.iterrows():
    product_data = pd.DataFrame([product_row])  # Create a single-row DataFrame
    analysis_result = agent.analyze_product(product_data)
    if analysis_result:  # Only append if analysis was successful (returned non-empty dict)
        results.append(analysis_result)
    break

# Create a DataFrame from the results
results_df = pd.DataFrame(results)
print(results_df.to_markdown(index=False))  # Output as Markdown table.  For CSV: results_df.to_csv("output.csv", index=False)


# To adapt to "Set 2 Seller Products", assuming it has a similar schema,  you can simply change the input dataframe:
# set2_products_df = pd.read_csv("set2_products.csv")
# ... (rest of the processing loop, using set2_products_df)
"""
