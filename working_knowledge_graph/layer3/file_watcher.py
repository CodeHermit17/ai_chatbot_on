# file_watcher.py

import os
import json
import hashlib
import time

# Import functions from the kg_extractor module
from kg_extractor import process_document_node

# --- Configuration ---
INPUT_DIRECTORY = "C:/Users/kashy/Documents/Code/Projects/ai_chatbot_on/working_knowledge_graph/layer2/processed" # <--- IMPORTANT: SET THIS TO YOUR DIRECTORY
MOSDAC_BASE_URL = "https://www.mosdac.gov.in/" # Define the base URL for shortening (using http as per your sample)

STATE_FILE = "C:/Users/kashy/Documents/Code/Projects/ai_chatbot_on/working_knowledge_graph/layer3/processed_files_state.json"
OUTPUT_KG_FILE = "C:/Users/kashy/Documents/Code/Projects/ai_chatbot_on/working_knowledge_graph/layer3/all_extracted_kg.json"

# --- Helper Functions ---
def load_processed_state():
    """Loads the state of processed files from a JSON file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {STATE_FILE} is corrupted. Starting with an empty state.")
            return {}
    return {}

def save_processed_state(state):
    """Saves the current state of processed files to a JSON file."""
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=4)

def calculate_file_hash(filepath):
    """Calculates the MD5 hash of a file's content."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""): # Read in chunks
            hasher.update(chunk)
    return hasher.hexdigest()

# --- Main Processing Logic ---
def process_new_or_modified_files(base_dir):
    """
    Scans the base directory for JSON files, processes new or modified ones,
    and updates a file to track changes.
    """
    processed_state = load_processed_state()
    all_extracted_knowledge = {} # To accumulate KG from all processed files in this run

    # Load existing KG results if any
    if os.path.exists(OUTPUT_KG_FILE):
        try:
            with open(OUTPUT_KG_FILE, 'r', encoding='utf-8') as f:
                all_extracted_knowledge = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {OUTPUT_KG_FILE} is corrupted or empty. Starting with an empty KG output.")
            all_extracted_knowledge = {}

    found_files_in_run = set() # Track files found in this run to remove old entries from state

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".json"): # Ensure case-insensitive check
                filepath = os.path.join(root, file)
                found_files_in_run.add(filepath)

                current_hash = calculate_file_hash(filepath)
                
                # Check if the file is new or its content has changed
                if filepath not in processed_state or processed_state[filepath].get('hash') != current_hash:
                    print(f"Processing: {filepath} (New or Modified)")
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                            
                            # Get the original_url from the top-level of the JSON
                            original_url_from_json = doc_data.get("original_url")
                            
                            doc_id_for_kg = os.path.splitext(file)[0] # Default to MD5 hash filename as ultimate fallback

                            if original_url_from_json:
                                if original_url_from_json.startswith(MOSDAC_BASE_URL):
                                    # Use the part after the base URL
                                    shortened_url_path = original_url_from_json[len(MOSDAC_BASE_URL):]
                                    if not shortened_url_path: # Handle the base URL itself (e.g., http://www.mosdac.gov.in/)
                                        doc_id_for_kg = "/"
                                    else:
                                        doc_id_for_kg = shortened_url_path
                                else:
                                    # If it's an external URL, use the full URL as doc_id
                                    doc_id_for_kg = original_url_from_json
                            else:
                                print(f"  Warning: 'original_url' not found at top-level in '{file}'. Using MD5 hash as doc_id in KG.")
                            
                            # IMPORTANT: Set the 'doc_id' in the loaded JSON data to the chosen KG ID
                            doc_data["doc_id"] = doc_id_for_kg

                            # Call the processing function from kg_extractor
                            triples = process_document_node(doc_data) 
                            
                            # Store triples indexed by the chosen doc_id
                            all_extracted_knowledge[doc_id_for_kg] = triples
                            
                            # Update the processed state with the new hash and the doc_id used in KG
                            processed_state[filepath] = {
                                'hash': current_hash,
                                'doc_id_in_kg': doc_id_for_kg # Store the doc_id used for this file in the KG
                            }
                            print(f"  Extracted {len(triples)} triples for Document ID: {doc_id_for_kg}.")
                            if original_url_from_json:
                                print(f"  (Original URL from JSON: {original_url_from_json})")
                            
                    except json.JSONDecodeError as e:
                        print(f"Error: Could not parse JSON file {filepath}. Error: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred while processing {filepath}: {e}")
                else:
                    print(f"Skipping: {filepath} (No changes detected)")
                    # If not processed, ensure its existing KG entry (if any) is retained
                    existing_doc_id_in_kg = processed_state[filepath].get('doc_id_in_kg')
                    if existing_doc_id_in_kg and existing_doc_id_in_kg not in all_extracted_knowledge:
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                doc_data = json.load(f)
                                doc_data["doc_id"] = existing_doc_id_in_kg # Use the stored doc_id
                                all_extracted_knowledge[existing_doc_id_in_kg] = process_document_node(doc_data)
                                print(f"  Re-added existing KG entry for Document ID: {existing_doc_id_in_kg}.")
                        except Exception as e:
                            print(f"Warning: Could not re-add KG for {filepath} with doc_id {existing_doc_id_in_kg}: {e}")


    # Clean up state file and KG for files that no longer exist
    # Collect all doc_ids that *should* be in the KG based on currently found files
    active_doc_ids_in_kg = set()
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".json"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                        # Get original_url from top-level of JSON for cleanup consistency
                        original_url_from_json = doc_data.get("original_url")
                        
                        if original_url_from_json:
                            if original_url_from_json.startswith(MOSDAC_BASE_URL):
                                shortened_url_path = original_url_from_json[len(MOSDAC_BASE_URL):]
                                if not shortened_url_path:
                                    active_doc_ids_in_kg.add("/")
                                else:
                                    active_doc_ids_in_kg.add(shortened_url_path)
                            else:
                                active_doc_ids_in_kg.add(original_url_from_json)
                        else:
                            active_doc_ids_in_kg.add(os.path.splitext(file)[0]) # Fallback to MD5 hash filename
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Warning: Could not read {filepath} for cleanup check: {e}")
                    pass


    files_to_remove_from_state = [f for f in processed_state if f not in found_files_in_run]
    for f in files_to_remove_from_state:
        print(f"Removing {f} from state: File no longer exists.")
        del processed_state[f]

    doc_ids_to_remove_from_kg = [doc_id for doc_id in all_extracted_knowledge if doc_id not in active_doc_ids_in_kg]
    for doc_id in doc_ids_to_remove_from_kg:
        print(f"Removing KG entry for Document ID: {doc_id} (original file no longer exists).")
        del all_extracted_knowledge[doc_id]

    save_processed_state(processed_state)
    
    # Save the accumulated knowledge graph
    with open(OUTPUT_KG_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_knowledge, f, indent=2)
    print(f"\nAll extracted knowledge graph data saved to {OUTPUT_KG_FILE}")

    return all_extracted_knowledge

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIRECTORY):
        print(f"Error: The input directory '{INPUT_DIRECTORY}' does not exist.")
        print("Please create this directory and place your JSON files inside it,")
        print("or update the 'INPUT_DIRECTORY' variable in this script.")
        exit()

    print(f"Starting JSON file watcher in '{INPUT_DIRECTORY}'...")
    
    # Run the processing
    processed_kg = process_new_or_modified_files(INPUT_DIRECTORY)
    print("\n--- Summary of current Knowledge Graph ---")
    print(f"Total documents processed/tracked: {len(processed_kg)}")
    
    # You can uncomment the following lines to run in a continuous loop for monitoring
    # while True:
    #     print(f"\nMonitoring '{INPUT_DIRECTORY}' for changes (next check in 60 seconds)...")
    #     processed_kg = process_new_or_modified_files(INPUT_DIRECTORY)
    #     time.sleep(60) # Wait for 60 seconds before checking again