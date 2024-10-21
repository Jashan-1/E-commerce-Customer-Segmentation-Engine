from rich import print
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def improve_query(user_query):
    """
    This function refines a user's raw query in English to make it clearer, more detailed, 
    and optimal for document retrieval systems. It uses the OpenAI API to generate a response based on the context 
    and query, enhancing clarity, removing ambiguity, and adding any missing context. The function ensures the 
    original intent of the question remains the same.

    Args:
        user_query (str): The user's raw query in English.

    Returns:
        str: The optimized query for document retrieval systems.
    """

    # Define the prompt template to ask the model to take parameters and return a product recommendation
    prompt_template = f"""
    ### Task Description:
    You are an AI assistant that specializes in making product recommendations based on user-provided parameters. 
    Below are the details related to a product review. Your task is to analyze the provided parameters and recommend 
    a suitable product based on the information.

    ### Instructions:
    1. Analyze the provided user query and input parameters (Rating, Title, Text, Images, ASIN, Parent ASIN, User ID, Timestamp, Helpful vote, Verified purchase).
    2. Based on this information, provide a relevant product recommendation.
    3. Make sure the product recommendation aligns with the preferences and context given in the review.

    ### Input Parameters:
    - Rating: {user_query['rating']}
    - Title: {user_query['title']}
    - Text: {user_query['text']}
    - Images: {user_query['images']}
    - ASIN: {user_query['asin']}
    - Parent ASIN: {user_query['parent_asin']}
    - User ID: {user_query['user_id']}
    - Timestamp: {user_query['timestamp']}
    - Helpful Vote: {user_query['helpful_vote']}
    - Verified Purchase: {user_query['verified_purchase']}

    ### Output:
    Please provide a product recommendation based on the above parameters. Only the product name is required, no explanations.
    """

    try:
        # Call the OpenAI API to generate the product recommendation
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Ensure to replace with the actual model as necessary
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_template}
            ],
            temperature=0.7,  # Adjust temperature for creativity (lower for precision)
            top_p=1,
            n=1
        )

        # Extract the product recommendation from the response
        product_recommendation = response.choices[0].message.content.strip()

        return product_recommendation,response

    except Exception as e:
        # Handle errors in the API call
        print(f"Error occurred while generating product recommendation: {e}")
        return "No recommendation available due to an error."  # Return a fallback message if there's an error
    

# Sample input data in English (user review)
user_query = {
    "rating": 4.5,
    "title": "Excellent product for face care.",
    "text": "This product makes my face smooth and shiny, with a great fragrance. I highly recommend it for those seeking a natural face care product.",
    "images": [],
    "asin": "B09XYZ123",
    "parent_asin": "B09XYZ123",
    "user_id": "USER12345",
    "timestamp": 1630456795000,
    "helpful_vote": 10,
    "verified_purchase": True
}

# Example function call
product_recommendation,response = improve_query(user_query)

# Output the recommendation
print("Recommended Product:", product_recommendation)
