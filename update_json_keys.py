import os
import json

# Directory containing the JSON files
directory = 'custom_workflows'

query_string = "realisticVisionV60B1_v60B1VAE.safetensors"
replacement_string = "juggernaut_reborn.safetensors"

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        filepath = os.path.join(directory, filename)

        
        # Read the JSON file
        with open(filepath, 'r') as file:
            data = file.read()

        # count the occurrences of the query string in the file:
        count = data.count(query_string)

        # Replace the specified string
        data = data.replace(query_string, replacement_string)

        # Write the modified content back to the file
        with open(filepath, 'w') as file:
            file.write(data)

        print(f"{count} replacements made in {filename}")
