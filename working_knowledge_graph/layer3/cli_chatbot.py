import sys
import os
import json
import re
import spacy
import requests # For making HTTP requests to the LLM API
import time # For potential delays between retries
from urllib.parse import urlparse # For robust URL parsing

# --- Configuration for kg_extractor.py path ---
# IMPORTANT: This path must point to the directory containing your kg_extractor.py file.
# Using a raw string (r"...") is good practice for Windows paths to avoid issues with backslashes.
KG_EXTRACTOR_DIR = r"C:\Users\kashy\Documents\Code\Projects\ai_chatbot_on\working_knowledge_graph\layer3"

# Add the directory to Python's system path if it's not already there
if KG_EXTRACTOR_DIR not in sys.path:
    sys.path.append(KG_EXTRACTOR_DIR)

# Now, import statements for kg_extractor will work
from kg_extractor import CANONICAL_ENTITIES, get_canonical_entity_info

# --- LLM API Configuration ---
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual Google Gemini API Key
API_KEY = "YOUR_API_KEY_HERE" 
LLM_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
MAX_LLM_RETRIES = 3 # Max attempts for LLM to generate a good response

class KnowledgeGraphChatbot:
    def __init__(self, kg_file_path):
        """
        Initializes the chatbot by loading the knowledge graph from a JSON file.
        """
        self.kg = self._load_knowledge_graph(kg_file_path)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spacy model 'en_core_web_sm'. Please wait...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.canonical_entity_map = self._build_canonical_entity_map()
        self.mosdac_core_entities = self._get_mosdac_core_entities() # For relevance check
        print(f"Knowledge Graph loaded with {len(self.kg)} documents/entities.")

    def _load_knowledge_graph(self, kg_file_path):
        """
        Loads the knowledge graph from the specified JSON file.
        """
        try:
            with open(kg_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Knowledge Graph file not found at {kg_file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {kg_file_path}. Check file format.")
            return {}

    def _build_canonical_entity_map(self):
        """
        Builds a comprehensive map from various entity mentions (from KG keys and values,
        and CANONICAL_ENTITIES) to their canonical forms.
        """
        canonical_map = {}
        
        for doc_id, triples in self.kg.items():
            canonical_map[doc_id.lower()] = doc_id
            if isinstance(doc_id, str):
                # Handle URL paths and query parameters for doc_ids
                parsed_id = urlparse(doc_id)
                if parsed_id.netloc: # If it's a full URL
                    # Add domain parts if relevant
                    domain_parts = parsed_id.netloc.split('.')
                    for part in domain_parts:
                        if part and part.lower() not in canonical_map:
                            canonical_map[part.lower()] = parsed_id.netloc
                
                # Add path segments
                path_parts = [p for p in parsed_id.path.split('/') if p]
                for part in path_parts:
                    if part.lower() not in canonical_map:
                        canonical_map[part.lower()] = doc_id # Map part to full doc_id
                
                # Add query parameters if they form meaningful entities
                if parsed_id.query:
                    for param in parsed_id.query.split('&'):
                        if '=' in param:
                            key, value = param.split('=', 1)
                            if key and key.lower() not in canonical_map:
                                canonical_map[key.lower()] = key
                            if value and value.lower() not in canonical_map:
                                canonical_map[value.lower()] = value
                        else:
                            if param and param.lower() not in canonical_map:
                                canonical_map[param.lower()] = param


            for s, p, o in triples:
                s_str = str(s)
                o_str = str(o)

                if s_str:
                    canonical_map[s_str.lower()] = s_str
                    for word in s_str.split():
                        if word.lower() not in canonical_map:
                            canonical_map[word.lower()] = s_str
                if o_str:
                    canonical_map[o_str.lower()] = o_str
                    for word in o_str.split():
                        if word.lower() not in canonical_map:
                            canonical_map[word.lower()] = o_str
        
        for phrase, info in CANONICAL_ENTITIES.items():
            canonical_map[phrase.lower()] = info["text"]
            for word in phrase.split():
                if word.lower() not in canonical_map:
                    canonical_map[word.lower()] = info["text"]

        return canonical_map

    def _get_mosdac_core_entities(self):
        """
        Builds a comprehensive set of entities that are considered "MOSDAC-related"
        by analyzing the knowledge graph.
        """
        mosdac_related_entities = set()

        # Explicitly add known MOSDAC-related terms
        mosdac_related_entities.add("mosdac")
        mosdac_related_entities.add("isro")
        mosdac_related_entities.add("space applications centre (sac)")
        mosdac_related_entities.add("sac")
        mosdac_related_entities.add("india")
        mosdac_related_entities.add("indian region")
        mosdac_related_entities.add("meteorological and oceanographic satellite data archival centre (mosdac)")
        mosdac_related_entities.add("/") # For the base URL doc_id

        # Iterate through all documents and triples to find related entities
        for doc_id, triples in self.kg.items():
            is_mosdac_doc = False
            # Check if the document ID itself is from MOSDAC domain or a core MOSDAC entity
            parsed_doc_id = urlparse(doc_id)
            if "mosdac.gov.in" in parsed_doc_id.netloc.lower() or doc_id.lower() in mosdac_related_entities:
                is_mosdac_doc = True
            
            # If it's a MOSDAC document, add its ID and all its subjects/objects
            if is_mosdac_doc:
                mosdac_related_entities.add(doc_id.lower())
                for s, p, o in triples:
                    mosdac_related_entities.add(str(s).lower())
                    mosdac_related_entities.add(str(o).lower())
                    # Also add individual words from multi-word entities if they are relevant
                    for word in str(s).split():
                        mosdac_related_entities.add(word.lower())
                    for word in str(o).split():
                        mosdac_related_entities.add(word.lower())
            else:
                # Even if not a MOSDAC doc, if a triple links to a MOSDAC entity,
                # then its subject/object might be relevant. (e.g., a product provided by MOSDAC)
                for s, p, o in triples:
                    if "mosdac.gov.in" in str(o).lower() or "mosdac.gov.in" in str(s).lower():
                        mosdac_related_entities.add(str(s).lower())
                        mosdac_related_entities.add(str(o).lower())
                        for word in str(s).split():
                            mosdac_related_entities.add(word.lower())
                        for word in str(o).split():
                            mosdac_related_entities.add(word.lower())
                    
                    # Special check for "provides" or "developed_by" relationships
                    # where the object is a known MOSDAC organization
                    if p in ["provides", "developed_by", "manages", "operates"]:
                        if str(o).lower() in ["mosdac", "isro", "space applications centre (sac)", "sac"]:
                            mosdac_related_entities.add(str(s).lower()) # Add the subject (e.g., product/satellite)

        return mosdac_related_entities


    def _is_query_relevant_to_mosdac(self, extracted_entities, query):
        """
        Checks if the user's query is relevant to the MOSDAC domain.
        """
        query_lower = query.lower()
        
        # Direct keyword checks
        if "mosdac" in query_lower or "isro" in query_lower or "sac" in query_lower or "mosdac.gov.in" in query_lower:
            return True

        # Check if any identified entity is in the comprehensive MOSDAC-related set
        for entity in extracted_entities:
            if entity.lower() in self.mosdac_core_entities:
                return True
            # Also check if any part of a multi-word entity matches a core entity
            for word in entity.split():
                if word.lower() in self.mosdac_core_entities:
                    return True
        
        # Fallback: If query contains general terms that might indicate MOSDAC relevance
        # but no specific entities were extracted (e.g., "What are the data products?").
        # This is a weak signal, use with caution.
        general_mosdac_terms = ["satellite", "data", "oceanography", "meteorological", "altimetry", "remote sensing", "earth observation"]
        if any(term in query_lower for term in general_mosdac_terms) and extracted_entities:
            # If general terms are present AND we extracted *some* entities,
            # it's likely relevant even if those entities aren't yet in mosdac_core_entities.
            # This helps for new entities that haven't been fully linked yet.
            return True


        return False


    def _extract_query_entities(self, query):
        """
        Extracts potential entities from the user's query using both direct matching
        against the canonical entity map and SpaCy's NER with our custom canonical entities.
        Returns a list of canonical entity names found in the query.
        """
        found_entities = set()
        query_lower = query.lower()
        doc = self.nlp(query)

        # 1. Direct matching against the pre-built canonical_entity_map (for KG-specific terms)
        sorted_map_keys = sorted(self.canonical_entity_map.keys(), key=len, reverse=True)
        for phrase_lower in sorted_map_keys:
            if re.search(r'\b' + re.escape(phrase_lower) + r'\b', query_lower):
                found_entities.add(self.canonical_entity_map[phrase_lower])

        # 2. SpaCy's NER and CANONICAL_ENTITIES from kg_extractor
        for ent in doc.ents:
            canonical_info = get_canonical_entity_info(ent.text)
            if canonical_info["type"] != "Unknown":
                found_entities.add(canonical_info["text"])
            else:
                if ent.label_ == "ORG" or ent.label_ == "GPE" or ent.label_ == "LOC" or ent.label_ == "PRODUCT" or ent.label_ == "DATE" or ent.label_ == "TIME":
                    found_entities.add(ent.text)

        for phrase, info in CANONICAL_ENTITIES.items():
            if re.search(r'\b' + re.escape(phrase.lower()) + r'\b', query_lower):
                found_entities.add(info["text"])

        return list(found_entities)


    def _find_relevant_triples(self, entities):
        """
        Finds all triples in the KG that involve any of the given entities.
        Ensures all triple elements are hashable (strings) before adding to the list.
        """
        relevant_triples = []
        entities_lower = {e.lower() for e in entities}

        for doc_id, triples in self.kg.items():
            doc_id_lower = doc_id.lower()
            doc_id_parts = [p.lower() for p in re.split(r'[/?#]', doc_id) if p]
            
            is_doc_id_relevant_to_entities = doc_id_lower in entities_lower or any(part in entities_lower for part in doc_id_parts)

            for s, p, o in triples:
                s_str = str(s)
                o_str = str(o)
                
                # Check if subject or object is one of the target entities
                if s_str.lower() in entities_lower or o_str.lower() in entities_lower:
                    relevant_triples.append((s_str, str(p), o_str))
                else:
                    for entity_text in entities:
                        entity_text_lower = str(entity_text).lower()
                        if entity_text_lower in s_str.lower() or entity_text_lower in o_str.lower():
                            relevant_triples.append((s_str, str(p), o_str))
                
                # If the doc_id itself was relevant, ensure its triples are added.
                if is_doc_id_relevant_to_entities and (s_str, str(p), o_str) not in relevant_triples:
                     relevant_triples.append((s_str, str(p), o_str))

        return list(set(relevant_triples)) # Return unique triples

    def _call_llm(self, prompt, response_schema=None):
        """
        Generic function to call the LLM API.
        """
        chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
        payload = {"contents": chat_history}
        
        if response_schema:
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }

        try:
            response = requests.post(LLM_API_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

            if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
                if response_schema:
                    try:
                        return json.loads(raw_text)
                    except json.JSONDecodeError:
                        # print(f"Warning: LLM returned non-JSON for structured response: {raw_text}")
                        return None # Indicate failure to parse structured response
                return raw_text
            else:
                # print(f"LLM response structure unexpected: {result}")
                return None
        except requests.exceptions.RequestException as e:
            # print(f"An error occurred while contacting the LLM: {e}")
            return None
        except Exception as e:
            # print(f"An unexpected error occurred during LLM call: {e}")
            return None

    def _generate_llm_response(self, query, relevant_triples, retry_reason=None):
        """
        Generates a natural language response using the Generator LLM.
        """
        if not relevant_triples:
            return "I don't have enough specific information in my knowledge graph to answer that directly."

        triples_str = "\n".join([f"({s}, {p}, {o})" for s, p, o in relevant_triples])

        comparison_keywords = ["difference", "compare", "vs", "versus"]
        is_comparison_query = any(keyword in query.lower() for keyword in comparison_keywords)

        llm_instruction = ""
        if is_comparison_query:
            llm_instruction = """
            Your task is to synthesize this information into a concise, natural, and user-friendly answer, specifically highlighting the **differences or comparisons** between the entities mentioned in the query based on the provided facts. If direct differences are not evident, state what each entity is or provides.
            """
        else:
            llm_instruction = """
            Your task is to synthesize this information into a concise, natural, and user-friendly answer.
            """
        
        if retry_reason:
            llm_instruction += f"\n\nPrevious attempt was not good because: '{retry_reason}'. Please improve your response based on this feedback."


        prompt = f"""
        You are a helpful AI assistant that answers questions based on a knowledge graph.
        I will provide you with a user's query and a list of relevant facts (triples) from my knowledge graph.
        {llm_instruction}

        User Query: "{query}"

        Relevant Facts (Triples):
        {triples_str}

        Instructions:
        1. Answer the user's query directly and concisely.
        2. Use only the information provided in the "Relevant Facts" section. Do not make up information.
        3. If the provided facts are insufficient to answer the query, state that you don't have enough information.
        4. Prioritize information directly related to the main entities in the query.
        5. Avoid listing raw triples. Convert them into natural language sentences.
        6. If the query asks "What does X provide?", focus on the "provides" relationships.
        7. If the query asks "What is X?", summarize its key attributes.
        8. Keep the response to 2-4 sentences if possible.

        Now, generate the response for the given query and facts:
        """
        return self._call_llm(prompt)

    def _testify_response(self, query, generated_response, relevant_triples):
        """
        Evaluates the generated response using a Testifier LLM.
        Returns a dictionary with 'status' (GOOD/BAD) and 'reason'.
        """
        if not generated_response:
            return {"status": "BAD", "reason": "Response was empty or failed to generate."}

        triples_str = "\n".join([f"({s}, {p}, {o})" for s, p, o in relevant_triples])

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "status": {"type": "STRING", "enum": ["GOOD", "BAD"]},
                "reason": {"type": "STRING", "description": "Reason if status is BAD, or 'N/A' if GOOD."}
            },
            "required": ["status", "reason"]
        }

        prompt = f"""
        You are an AI assistant whose job is to evaluate the quality of another AI's response.
        I will provide you with a User Query, the Relevant Facts (triples) from a knowledge graph,
        and a Generated Response.

        Your task is to determine if the Generated Response is GOOD or BAD based on these criteria:
        - **Conciseness:** Is the response brief (ideally 2-4 sentences) and to the point?
        - **Accuracy:** Does the response accurately reflect *only* the information in the Relevant Facts? Does it avoid hallucination?
        - **Directness:** Does it directly answer the User Query?
        - **Readability:** Is it natural-sounding and easy to understand?
        - **Completeness (within facts):** Does it utilize the most important relevant facts to answer the query?

        If the User Query asks for a "difference" or "comparison", the response should *attempt* to highlight differences if the facts allow, or state what each entity is/does.

        Return your evaluation as a JSON object with a 'status' (GOOD/BAD) and a 'reason' (if BAD, explain why; if GOOD, state 'N/A').

        User Query: "{query}"

        Relevant Facts (Triples):
        {triples_str}

        Generated Response:
        {generated_response}

        Your evaluation (JSON format):
        """
        
        evaluation_result = self._call_llm(prompt, response_schema=response_schema)
        
        if evaluation_result and isinstance(evaluation_result, dict):
            return evaluation_result
        else:
            return {"status": "BAD", "reason": "Testifier LLM failed to return valid structured response or API call failed."}


    def answer_query(self, query):
        """
        Answers a user query using the loaded knowledge graph and an LLM for response generation,
        with a self-correction loop.
        """
        extracted_entities = self._extract_query_entities(query)

        # 1. Relevance Check
        if not self._is_query_relevant_to_mosdac(extracted_entities, query):
            return "I can only answer questions related to the MOSDAC website and its associated entities (satellites, products, organizations, etc.). Please ask a relevant query."

        if not extracted_entities:
            return "I couldn't identify any specific entities or topics in your query. Please try rephrasing or be more specific. For example, ask about 'INSAT-3D', 'Rainfall Estimate', or 'MOSDAC'."

        relevant_triples = self._find_relevant_triples(extracted_entities)

        if not relevant_triples:
            # If no direct triples, try to find entities of the same TYPE for a general answer
            found_types = {get_canonical_entity_info(e)["type"] for e in extracted_entities if get_canonical_entity_info(e)["type"] != "Unknown"}
            
            if found_types:
                general_info_triples = []
                for doc_id, triples in self.kg.items():
                    for s, p, o in triples:
                        s_info = get_canonical_entity_info(str(s))
                        o_info = get_canonical_entity_info(str(o))
                        if s_info["type"] in found_types or o_info["type"] in found_types:
                            general_info_triples.append((str(s), str(p), str(o)))
                            if len(general_info_triples) > 10:
                                break
                    if len(general_info_triples) > 10:
                        break
                
                if general_info_triples:
                    relevant_triples = general_info_triples # Use these for LLM generation
                else:
                    return f"I found entities like {', '.join(extracted_entities)}, but no direct information related to them in my knowledge base. I also couldn't find general information about these types."
            else:
                return f"I found entities like {', '.join(extracted_entities)}, but no direct information related to them in my knowledge base."

        # 2. Self-Correction Loop for LLM Response Generation
        final_response = "I'm sorry, I couldn't generate a good response after multiple attempts."
        retry_reason = None

        for attempt in range(MAX_LLM_RETRIES):
            generated_response = self._generate_llm_response(query, relevant_triples, retry_reason)
            
            if not generated_response:
                retry_reason = "Generator LLM failed to produce any response."
                time.sleep(1) # Small delay before retrying
                continue

            evaluation = self._testify_response(query, generated_response, relevant_triples)
            
            if evaluation["status"] == "GOOD":
                final_response = generated_response
                break
            else:
                retry_reason = evaluation["reason"]
                time.sleep(1) # Small delay before retrying

        return final_response

# --- Main execution block for testing the chatbot ---
if __name__ == "__main__":
    kg_file = "C:/Users/kashy/Documents/Code/Projects/ai_chatbot/all_extracted_kg.json" # Make sure this file is in the same directory as this script

    chatbot = KnowledgeGraphChatbot(kg_file)

    print("\n--- Knowledge Graph Chatbot ---")
    print("Type your queries (e.g., 'What is insitu?', 'Tell me about SARAL-AltiKa', 'What does MOSDAC provide?').")
    print("Try queries that don't have direct keywords, like 'What are the satellites?' or 'Tell me about data products'.")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour query: ")
        if user_query.lower() == 'exit':
            print("Exiting chatbot. Goodbye!")
            break
        
        response = chatbot.answer_query(user_query)
        print("\nChatbot Response:")
        print(response)
