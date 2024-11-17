import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # Updated import

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChatOpenAI model with the fine-tuned model
llm = ChatOpenAI(
    model="ft:gpt-3.5-turbo-1106:personal:bio-generator:AURS8hvs",  # Use fine-tuned model
    temperature=0.7,
    max_tokens=150,
    openai_api_key=openai_api_key,  # Pass the key
)

# Define the Prompt Template
template = """Create a personalized bio for a user with the following details:
Career: {career},
Personality Traits: {personality_traits},
Interests: {interests},
Relationship Goals: {relationship_goals}.
"""

prompt = PromptTemplate(
    input_variables=["career", "personality_traits", "interests", "relationship_goals"],
    template=template,
)

# Pydantic model for the request body
class BioRequest(BaseModel):
    career: str
    personality: List[str]
    interests: List[str]
    relationship_goals: str

# Define the FastAPI app
app = FastAPI()

def generate_bio(user_input):
    """
    Generates a personalized bio based on user input.
    """
    # Fill the template with user inputs
    filled_prompt = prompt.format(
        career=user_input["career"],
        personality_traits=", ".join(user_input["personality"]),
        interests=", ".join(user_input["interests"]),
        relationship_goals=user_input["relationship_goals"],
    )
    
    # Generate the bio using the fine-tuned model
    response = llm.predict(filled_prompt)  # Use the fine-tuned model
    return response.strip()

@app.post("/generate_bio/")
async def generate_bio_endpoint(bio_request: BioRequest):
    """
    API endpoint to generate a personalized bio.
    """
    try:
        # Prepare user input for bio generation
        user_input = {
            "career": bio_request.career,
            "personality": bio_request.personality,
            "interests": bio_request.interests,
            "relationship_goals": bio_request.relationship_goals,
        }
        
        # Call the generate_bio function
        bio = generate_bio(user_input)
        
        # Return the generated bio
        return {"generated_bio": bio}
    
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"Error generating bio: {e}")