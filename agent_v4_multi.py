# Multi agent system with retrival code, Toolagent and Langfuse

# Use best practices for multi-agent systems, including clear communication protocols, 
# defined roles, and efficient resource management. 
# Ensure that each agent has a specific task and can interact with other agents effectively.

# Use best practices for General Python Development!

# Tracing
LangFuse: LANGFUSE_SECRET_KEY="sk-lf-2a24bee0-e7ca-4f88-8d9e-3d511708d502"
LANGFUSE_PUBLIC_KEY="pk-lf-ccc486e0-0d3c-4245-9f5f-987353d79cea"
LANGFUSE_BASE_URL="https://cloud.langfuse.com"

# Model & Agent Setup
OMLX_API_KEY = "sk-omlx-gWACPPkjsP67zAEgJ3GReJ6q"
OMLX_BASE_URL = "http://localhost:8000/v1" 
MODEL_ID = "Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-mlx-4Bit"

# Description of the task:
# 1. create a dataset with 1000 questions and answer about physio related questions around muscles, tendons and nerve issues that are not obvious to test how many my model gets correct. Possibly Use an external dataset easily available online for this.
# 2. The model used as multi agent is Qwen3.5-9B-Claude-4.6-HighIQ-INSTRUCT-HERETIC-UNCENSORED-mlx-4Bit - its a small model so we need to use all the tricks to make it be able to solve compelx tasks
# 3. I want the medical agent to grill the patient and make him perform tests and descripe sympthoms until the model is certain to diagnose
# 4. The model should be able to ask for more information if it is not certain about the diagnosis
# 5. The model should be able to search and do so frequently to gain more knowledge and improve its accuracy
# 6. The model should be able to store information localy about a given interaction and retrieve it later if needed.
# 7. The tool should use COnsolidate context, store intermiedient results and use them to improve the final answer while impriving context efficiency
# 8. The model should perform offline test on the intial dataset of 1000 medical questions or other source to test accuracy in Langfuse. Note that this test set is one shot and cannot grill the user for more input




