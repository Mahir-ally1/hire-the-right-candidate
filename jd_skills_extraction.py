## Setup

import boto3
from botocore.exceptions import ClientError
from nano_graphrag import GraphRAG, QueryParam

model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0" # cheap
# model = "us.meta.llama3-2-3b-instruct-v1:0"

## Looking at job description and extracting key skills

job_description_example = '''
- Develop and deploy machine learning models for various applications.
- Collaborate with cross-functional teams to understand business requirements and create solutions.
- Conduct research to stay current on advancements in machine learning and artificial intelligence.
- Optimize algorithms for scalability, efficiency, and accuracy.
- Work on data preprocessing, feature engineering, and model evaluation.
- Mentor junior team members and provide technical guidance.
- Communicate complex technical concepts to non-technical stakeholders.
- Contribute to the overall architecture and design of machine learning systems.
- Implement best practices for data management and governance.
- Participate in code reviews and provide constructive feedback.
- Experience with big data technologies such as Hadoop, Spark, or Hive.
- Knowledge of cloud platforms like AWS, Azure, or Google Cloud.
- Publications or contributions to the machine learning community.
'''

# Create an Amazon Bedrock Runtime client.
brt = boto3.client("bedrock-runtime")

# Start a conversation with the user message.
user_message = f'''
You are a recruiter, and your job is to find the key skills in the <job_description> tags. 
You will respond with the key softwares, skills in a comma seperated list.
Be as specific as you can.

<job_description>
{job_description_example}
</job_description>

'''
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

try:
    # Send the message to the model, using a basic inference configuration.
    response = brt.converse(
        modelId=model_id,
        messages=conversation,
        inferenceConfig={"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    )

    # Extract and print the response text.
    response_text = response["output"]["message"]["content"][0]["text"]
    jd_key_skills = response_text
    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)