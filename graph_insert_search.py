## Setup

from nano_graphrag import GraphRAG, QueryParam

model = "us.anthropic.claude-3-5-haiku-20241022-v1:0" # cheap
# model = "us.meta.llama3-2-3b-instruct-v1:0"


## Constructing GraphRAG and retrieving skills from knowledge graph

graph_func = GraphRAG(
    working_dir="/Users/mahirjain/Desktop/Personal Projects/matchmind-included-infoleague/skills",
    using_amazon_bedrock=True,
    best_model_id=model,
    cheap_model_id=model,
)

with open("/Users/mahirjain/Desktop/Personal Projects/matchmind-included-infoleague/nano-graphrag/tests/mock_data.txt") as f:
    graph_func.insert(f.read())


prompt = "Respond as a comma seperated list only, skills for a software developer?"

# Perform global graphrag search
print(graph_func.query(prompt, param=QueryParam(mode="global")))

# Perform local graphrag search (I think is better and more scalable one)
print(graph_func.query(prompt, param=QueryParam(mode="local")))