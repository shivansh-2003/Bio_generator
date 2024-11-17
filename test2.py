import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChatOpenAI model with your fine-tuned model
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
Hobbies: {hobbies},
Relationship Goals: {relationship_goals}.
"""

prompt = PromptTemplate(
    input_variables=["career", "personality_traits", "hobbies", "relationship_goals"],
    template=template,
)

def generate_bio(user_input):
    # Fill the template with user inputs
    filled_prompt = prompt.format(
        career=user_input["career"],
        personality_traits=", ".join(user_input["personality"]),
        hobbies=", ".join(user_input["hobbies"]),
        relationship_goals=user_input["relationship_goals"],
    )
    
    # Generate the bio using the fine-tuned model
    response = llm.predict(filled_prompt)  # Use the fine-tuned model
    return response.strip()

# Example Input
user_input = {
    "career": "Chef",
    "personality": ["Creative", "Outgoing"],
    "hobbies": ["Cooking", "Traveling"],
    "relationship_goals": "Adventurous",
}

# Generate and print the bio
print(generate_bio(user_input))