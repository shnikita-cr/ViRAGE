import datetime
import os

import ollama


def analyze_project_with_ollama(artifact_folder_path, readme_path, results_folder):
    """
    Analyzes multiple project files based on their readme and artifact files using Ollama,
    saving the responses to markdown files in a results folder.

    Args:
        artifact_folder_path (str): Path to the folder containing project artifacts.
        readme_path (str): Path to the project readme file.
        results_folder (str): Path to the folder where markdown files will be saved.
    """

    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Iterate through all files in the artifact folder
    for filename in os.listdir(artifact_folder_path):
        if filename.endswith(".md") or filename.endswith(".json"):  # Adjust file extensions if needed
            file_path = os.path.join(artifact_folder_path, filename)

            try:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
            except FileNotFoundError:
                print(f"Error: Readme file not found at {readme_path}")
                continue  # Skip to the next file

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    artifact_content = f.read()
            except FileNotFoundError:
                print(f"Error: Artifact file not found at {file_path}")
                continue  # Skip to the next file

            # Construct the prompt.
            prompt = f"""
            You are a project analysis assistant.  Your goal is to assess the project's health,
            identify potential problems, and suggest improvements based on the following information:

            Readme:
            {readme_content}

            Artifact from project step:
            {artifact_content}

            Respond with a concise summary of your analysis and recommendations.
            Focus on potential weaknesses and things to improve/clarify
            """

            # Interact with Ollama
            ollama_client = ollama.Client()
            response = ollama_client.generate(
                model='gemma3:4b',  # Or your preferred Ollama model
                prompt=prompt,
                stream=False,
                # think=True
            )

            # Generate filename for the response markdown file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ollama_response_{timestamp}_{filename}.md"  # Include filename in filename
            filepath = os.path.join(results_folder, filename)

            # Save the response to a markdown file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.response)

            print(f"Ollama Response saved to: {filepath}")


if __name__ == "__main__":
    artifact_folder = "input_folder/nodes"
    readme_file = "../../../README.md"
    results_folder = "output_folder"

    analyze_project_with_ollama(artifact_folder, readme_file, results_folder)
