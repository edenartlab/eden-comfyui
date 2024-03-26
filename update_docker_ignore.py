import os
import json

workflow_directory = 'custom_workflows'
model_folder       = 'ComfyUI/models'
model_extensions   = ['.pth', '.pt', '.safetensors', '.ckpt', '.onnx', '.bin']
dockerignore_file  = '.dockerignore'

models_to_include  = []

def find_model_name_in_workflow(workflow_data, model_extension):
    """
    loop over all the values in the dictionary and recursively search for the model_extension
    """
    models = []
    if type(workflow_data) is not dict:
        if type(workflow_data) is list:
            for item in workflow_data:
                models.extend(find_model_name_in_workflow(item, model_extension))
        elif type(workflow_data) is str and workflow_data.endswith(model_extension):
            models.append(workflow_data)
        return models

    for key, value in workflow_data.items():
        if isinstance(value, dict):
            models.extend(find_model_name_in_workflow(value, model_extension))
        elif isinstance(value, list):
            for item in value:
                models.extend(find_model_name_in_workflow(item, model_extension))
        elif isinstance(value, str) and value.endswith(model_extension):
            models.append(value)
    return models

# Iterate over each file in the workflow_directory
for filename in os.listdir(workflow_directory):
    if filename.endswith('_api.json'):
        filepath = os.path.join(workflow_directory, filename)

        # Read the JSON file
        with open(filepath, 'r') as file:
            data = file.read()

        # Parse the JSON file
        workflow_data = json.loads(data)

        # recursively search for models in the workflow_data:
        for model_extension in model_extensions:
            models_to_include.extend(find_model_name_in_workflow(workflow_data, model_extension))
            models_to_include = list(set(models_to_include))

# now that we have all the model names, lets find their absolute paths relative to the model_folder:
model_paths = []
for model in models_to_include:
    # search for the model in the model_folder:
    for root, dirs, files in os.walk(model_folder):
        if model in files:
            model_paths.append(os.path.join(root, model))

model_paths = sorted(model_paths)
for model_path in model_paths:
    print(model_path)

# Load the existing .dockerignore file:
with open(dockerignore_file, 'r') as file:
    dockerignore_data = file.read()

print(dockerignore_data)

delimiter_line = "### Include pipeline models: ###"
# Delete everything after the delimiter_line:
dockerignore_data = dockerignore_data.split(delimiter_line)[0]

# Add the delimiter_line:
dockerignore_data += delimiter_line + '\n'

# Add the model paths:
for model_path in model_paths:
    dockerignore_data += '!' + model_path + '\n'

# Write the updated .dockerignore file:
with open(dockerignore_file, 'w') as file:
    file.write(dockerignore_data)