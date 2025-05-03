import os
import fitz  # PyMuPDF
import pandas as pd
import boto3
import json
import re
import time 
from datetime import datetime



timestr = time.strftime("%Y%m%d-%H%M%S")
#model = "us.anthropic.claude-3-5-haiku-20241022-v1:0" # cheap
model = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
# Set up Bedrock client
bedrock = boto3.client("bedrock-runtime")

# try:
#     response = bedrock.invoke_model(
#         modelId=model,
#         contentType="application/json",
#         accept="application/json",
#         body=json.dumps({
#             "anthropic_version": "bedrock-2023-05-31",
#             "messages": [
#                 {"role": "user", "content": "Hello, Bedrock!"}
#             ],
#             "max_tokens": 5
#         })
#     )
#     # Read and decode the response body
#     result = json.loads(response['body'].read())
#     print("Bedrock runtime is live! Model response:", result)
# except Exception as e:
#     print("Bedrock runtime is NOT live or accessible. Error:", e)




# Set up the job description
job_description = f""" 

Responsibilities

The SCOT Labs team within Forecasting and Labs is responsible for designing and executing the inference and experimentation systems that measure the impact of SCOT initiatives. We are looking for senior applied scientists to drive innovation in SCOT by developing/building a new scientific approach and pushing our system further upstream in the innovation process. Key responsibilities of a Research Scientist in IPC Lab include:

 Developing new statistical, causal, and machine learning techniques and develop solution prototypes to drive innovation
 Working with technical and non-technical customers to design experiments and communicate results
 Collaborating with our dedicated software team to create production implementations for large-scale data analysis
 Developing an understanding of key business metrics / KPIs and providing clear, compelling analysis that shapes the direction of our business
 Presenting research results to our internal research community
 Leading training and informational sessions on our science and capabilities
 Your contributions will be seen and recognized broadly within Amazon, contributing to the Amazon research corpus and patent portfolio.

To help describe some of our challenges, we created a short video about at Amazon - http://bit.ly/amazon-scot

Amazon is an Equal Opportunity-Affirmative Action Employer – Minority / Female / Disability / Veteran / Gender Identity / Sexual Orientation / Age

Basic Qualifications

 3+ years of data querying languages (e.g. SQL), scripting languages (e.g. Python) or statistical/mathematical software (e.g. R, SAS, Matlab, etc.) experience
 3+ years of data scientist experience
 3+ years of machine learning/statistical modeling data analysis tools and techniques, and parameters that affect their performance experience
 Bachelor's degree
 Experience applying theoretical models in an applied environment

Preferred Qualifications

 Experience in Python, Perl, or another scripting language
 Experience in a ML or data scientist role with a large technology company 
"""

# Standard entity and relation types
# These are the standard entity and relation types that should always be included in the output
standard_entity_types = {
    "skill": "A technical skill possessed by the candidate, e.g., 'Python', 'SQL', 'Tableau'",
    "job_title": "A specific role or position held by the candidate, e.g., 'Data Scientist', 'Software Engineer'",
    "company": "An organization where the candidate has worked",
    "industry": "The domain or sector associated with a company or job, e.g., 'Finance', 'Healthcare'",
    "education": "The highest level of education held by the candidate, e.g. 'Masters', 'Bachelors', 'PhD'",
    "university": "The name of the educational institution, e.g., 'University of Washington'",
    "certification": "Professional certification or license held, e.g., 'AWS Certified Data Analyst'",
    "project": "Notable project or research the candidate has worked on",
    "soft_skill": "Behavioral or interpersonal skill, e.g., 'leadership', 'teamwork', 'adaptability'",
    "experience_length": "Duration of work experience measured in years"
}
standard_relation_types = {
    "hasSkill": "Candidate possesses this skill",
    "hasJobTitle": "Candidate has held this job title",
    "workedAt": "Candidate has worked at this company",
    "hasIndustry": "Candidate has worked in this industry",
    "studiedAt": "Candidate studied at this university",
    "certifiedIn": "Candidate holds this certification",
    "hasProject": "Candidate completed this project",
    "hasSoftSkill": "Candidate possesses this soft skill",
    "hasExperienceLength": "Candidate has this length of work experience",
    "hasEducationLevel": "Candidate attained this education level",
}


# This function calls the LLM with a schema prompt and returns entity_types and relation_types as Python dicts.
def extract_schema_from_job_description(job_description, standard_entity_types, standard_relation_types):
    """
    Calls the LLM with a schema prompt and returns entity_types and relation_types as Python dicts.
    """
    schema_prompt = f"""
    You are an expert in information extraction and knowledge graph construction.
    Given the following job description, identify all relevant entity types and relationship types that should be extracted from candidate resumes to match this job.

    **Always include all of the standard entity types and relationship types below in your output.**
    If you identify new types from the job description, **add them to the standard lists** (do not remove or replace the standard types).

    Standard_entity_types: {json.dumps(standard_entity_types)}
    Standard relation_types: {json.dumps(standard_relation_types)}

    Besides the standard entity and relation types, you may also create new types based on the job description.
    - For each entity type, provide a key and a short description.
    - For each relationship type, provide a key and a short description.
    - If the job description specifies a domain-specific requirement (e.g., "2 years of experience in data field"), create a specific relationship type (e.g., "hasDataExperience": "Years of experience in data-related roles").
    - Output two JSON objects: one for entity_types, one for relation_types. **Each should contain both the standard types and any new types you identify.**

    If the job description specifies a domain-specific requirement, create new types as needed.

    Job Description:
    \"\"\"
    {job_description}
    \"\"\"

    Respond with:
    entity_types = {{ ... }}
    relation_types = {{ ... }}
    """
    response = call_claude_bedrock(schema_prompt)
    # Extract the JSON objects from the response
    entity_types = {}
    relation_types = {}
    # Use regex to extract the JSON objects
    import re
    entity_match = re.search(r"entity_types\s*=\s*({.*?})\s*relation_types", response, re.DOTALL)
    relation_match = re.search(r"relation_types\s*=\s*({.*})", response, re.DOTALL)
    if entity_match:
        try:
            entity_types = json.loads(entity_match.group(1))
        except Exception as e:
            print("Error parsing entity_types JSON:", e)
    if relation_match:
        try:
            relation_types = json.loads(relation_match.group(1))
        except Exception as e:
            print("Error parsing relation_types JSON:", e)
    return entity_types, relation_types


def call_claude_bedrock(prompt):
    response = bedrock.invoke_model(
        modelId=model,  # or whichever version you have access to
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 7000,
            "temperature": 0.2
        })
    )
    result = json.loads(response['body'].read())
    return result['content'][0]['text']


entity_types, relation_types = extract_schema_from_job_description(
    job_description, standard_entity_types, standard_relation_types
)
print(json.dumps(entity_types, indent=2))
print(json.dumps(relation_types, indent=2))