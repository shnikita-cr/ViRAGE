import os

import ollama


def summarize_folder_content(input_folder, output_file):
    """
    Summarizes the entire content of a folder and saves the combined summary
    to a single markdown file.

    Args:
        input_folder (str): Path to the folder containing the files to summarize.
        output_file (str): Path to the output markdown file where the summary will be saved.
    """

    all_content = ""

    # Iterate through all files in the folder and its subfolders
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_content += f.read() + "\n\n"  # Append content and a separator
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Construct the prompt for Ollama - summarizing the entire folder content
    prompt = f"""
    Summarize the following content from a folder:
    {all_content}
    Provide a concise and informative summary of the entire folder's content.
    Focus on potential weaknesses and things to improve/clarify
    Output must be structured.
    """

    # Interact with Ollama
    ollama_client = ollama.Client()
    response = ollama_client.generate(
        model="gemma3:4b",
        prompt=prompt,
        stream=False,
    )

    # Save the response to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Summary of Folder Content\n\n")
        f.write(response.response)

    print(f"Summary saved to: {output_file}")


if __name__ == "__main__":
    input_folder = "output_folder"
    output_file = "summary.md"
    summarize_folder_content(input_folder, output_file)
