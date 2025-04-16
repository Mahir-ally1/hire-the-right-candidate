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


entity_types = {
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

relation_types = {
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


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

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


def get_candidate_data_from_text(text, candidate_id):
    try:
        prompt = f"""


You are an expert resume parser and your job is to extract entities and relations from the provided resume. 

Only extract from the following list of entities 

{entity_types}

with their associated relations 

{relation_types}

Extract these values:

- candidate name
- relation_type
- entity_type
- entity_value
- CANDIDATE_ID (use {candidate_id})
- HEADLINE or TITLE in a maximum of 5 words
- SUMMARY or ABOUT in a maximum of 20 words


Structure the output only as a list of dictionaries with one dictionary per relationship
Make sure to extract atleast 5 entities with their associated relations per candidate
Extract the output in the following format only as shown in <example_output> tags
Respond strictly only with the list without any other special characters, tags, or explanation

<example_output>
[
    {{
        "candidate_id": 1925202,
        "name": "Mahir Jain",
        "relationship": "hasSkill",
        "entity_type": "skill",
        "entity_value": "SQL",
        "TITLE": "Experienced Data Scientist",
        "SUMMARY": "Experienced data analyst with optimization expertise in automotive industry, pursuing MS in Information Management with strong technical skills.""
    }},
    {{
       "candidate_id": 1925202,
        "name": "Mahir Jain",
        "relationship": "workedAt",
        "entity_type": "company",
        "entity_value": "Cox Communications",
        "TITLE": "Experienced Data Scientist",
        "SUMMARY": "Experienced data professional with skills in SQL, Python, R, and data visualization tools. Background in sales operations and customer relationship management.""
    }}
]
</example_output>
Resume text:
{text}
"""
        data = call_claude_bedrock(prompt)
        with open(f"claude_output_candidate_{candidate_id}.txt", "w", encoding="utf-8") as f:
            f.write(data)
        # data = json.loads(extract_json_from_output_tag(raw_output))
        return data
    except Exception as e:
        print(f"Error processing candidate {candidate_id}: {e}")
        return []

def process_resumes_in_folder(folder_path):
    all_candidates_data = []
    candidate_id = 1
    extracted_entities = ''

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            print(f"Processing: {filename}")
            pdf_path = os.path.join(folder_path, filename)
            text= extract_text_from_pdf(pdf_path)
            extracted_entities+=get_candidate_data_from_text(text,candidate_id)
            candidate_id+=1


    return extracted_entities



# Example usage:
candidates = process_resumes_in_folder("Resumes") # Update where your resumes are stored

timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
with open(f'candidates_{timestr}.txt', 'w', encoding='utf-8') as f:
    f.write(candidates)