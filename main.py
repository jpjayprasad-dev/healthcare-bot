import os
import pdfplumber
from pathlib import Path
import pandas as pd
from operator import itemgetter
import json
import tiktoken
import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from sentence_transformers import CrossEncoder, util

openai.api_key = os.getenv('OPENAI_KEY') if os.getenv('OPENAI_KEY') else open("open_ai_key.txt", "r").read().strip("\n")

chroma_data_path = 'data/ChromaDB_Data'
chroma_client = chromadb.PersistentClient(path=chroma_data_path)

embedding_model = "text-embedding-ada-002"
embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=embedding_model)

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def check_bboxes(word, table_bbox):
    # Check whether word is inside a table bbox.
    l = word['x0'], word['top'], word['x1'], word['bottom']
    r = table_bbox
    return l[0] > r[0] and l[1] > r[1] and l[2] < r[2] and l[3] < r[3]

def extract_tables_from_pdf(pdf):
    p = 0
    full_text = []
    for page in pdf.pages:
        page_no = f"Page {p+1}"
        text = page.extract_text()

        tables = page.find_tables()
        
        table_bboxes = [i.bbox for i in tables]
        tables = [{'table': i.extract(), 'top': i.bbox[1]} for i in tables]
        non_table_words = [word for word in page.extract_words() if not any(
            [check_bboxes(word, table_bbox) for table_bbox in table_bboxes])]
        lines = []
        for cluster in pdfplumber.utils.cluster_objects(non_table_words + tables, itemgetter('top'), tolerance=5):

            if 'table' in cluster[0]:
                lines.append(json.dumps(cluster[0]['table']))

        if lines:
            full_text.append([page_no, " ".join(lines)])
        p +=1
    return full_text

def process_pdf(pdf_path):
    # Process the PDF file
    print(f"...Processing {pdf_path}")
    pdf = pdfplumber.open(pdf_path)

    # Call the function to extract the text from the PDF
    extracted_text = extract_tables_from_pdf(pdf)

    # Convert the extracted list to a PDF, and add a column to store document names
    extracted_text_df = pd.DataFrame(extracted_text, columns=['Page No.', 'Page_Text'])
    extracted_text_df['Document Name'] = pdf_path
    extracted_text_df['Metadata'] = extracted_text_df.apply(lambda x: {'Document_Name': x['Document Name'][:-4], 'Page_No.': x['Page No.']}, axis=1)

    # Print a message to indicate progress
    print(f"Finished processing {pdf_path}")

    return extracted_text_df

def store_vector(extracted_text_df):
    collection = chroma_client.get_or_create_collection(name='Lab_Reports', embedding_function=embedding_function)
    # Convert the page text and metadata from your dataframe to lists to be able to pass it to chroma

    documents_list = extracted_text_df["Page_Text"].tolist()
    metadata_list = extracted_text_df['Metadata'].tolist()

    # Add the documents and metadata to the collection alongwith generic integer IDs. You can also feed the metadata information as IDs by combining the policy name and page no.

    collection.add(
        documents= documents_list,
        ids = [str(i) for i in range(0, len(documents_list))],
        metadatas = metadata_list
    )
    cache_collection = chroma_client.get_or_create_collection(name='Lab_Reports_Cache', embedding_function=embedding_function)
    return collection, cache_collection

def run_query(query, extracted_text_df):
    collection, cache_collection = store_vector(extracted_text_df)
    # Searh the Cache collection first
    # Query the collection against the user query and return the top 20 results

    cache_results = cache_collection.query(
        query_texts=query,
        n_results=1
    )

    # Implementing Cache in Semantic Search

    # Set a threshold for cache search
    threshold = 0.2

    ids = []
    documents = []
    distances = []
    metadatas = []
    results_df = pd.DataFrame()


    # If the distance is greater than the threshold, then return the results from the main collection.

    if cache_results['distances'][0] == [] or cache_results['distances'][0][0] > threshold:
        # Query the collection against the user query and return the top 10 results
        results = collection.query(
        query_texts=query,
        n_results=10
        )

        # Store the query in cache_collection as document w.r.t to ChromaDB so that it can be embedded and searched against later
        # Store retrieved text, ids, distances and metadatas in cache_collection as metadatas, so that they can be fetched easily if a query indeed matches to a query in cache
        Keys = []
        Values = []
        for key, val in results.items():
            if key != 'embeddings' and val:
                for i in range(len(results['ids'][0])):
                    Keys.append(str(key)+str(i))
                    Values.append(str(val[0][i]))


        cache_collection.add(
            documents= [query],
            ids = [query],  # Or if you want to assign integers as IDs 0,1,2,.., then you can use "len(cache_results['documents'])" as will return the no. of queries currently in the cache and assign the next digit to the new query."
            metadatas = dict(zip(Keys, Values))
        )

        print("Not found in cache. Found in main collection.")

        result_dict = {'Metadatas': results['metadatas'][0], 'Documents': results['documents'][0], 'Distances': results['distances'][0], "IDs":results["ids"][0]}
        results_df = pd.DataFrame.from_dict(result_dict)

    # If the distance is, however, less than the threshold, you can return the results from cache

    elif cache_results['distances'][0][0] <= threshold:
        cache_result_dict = cache_results['metadatas'][0][0]

        # Loop through each inner list and then through the dictionary
        for key, value in cache_result_dict.items():
            if 'ids' in key:
                ids.append(value)
            elif 'documents' in key:
                documents.append(value)
            elif 'distances' in key:
                distances.append(value)
            elif 'metadatas' in key:
                metadatas.append(value)

        print("Found in cache!")

        # Create a DataFrame
        results_df = pd.DataFrame({
            'IDs': ids,
            'Documents': documents,
            'Distances': distances,
            'Metadatas': metadatas
        })

    top_3_RAG = rank(query, results_df)
    response = generate_response(query, top_3_RAG)
    return response

def rank(query, results_df):
    # Input (query, response) pairs for each of the top 20 responses received from the semantic search to the cross encoder
    # Generate the cross_encoder scores for these pairs

    cross_inputs = [[query, response] for response in results_df['Documents']]
    cross_rerank_scores = cross_encoder.predict(cross_inputs)
    results_df['Reranked_scores'] = cross_rerank_scores

    top_3_rerank = results_df.sort_values(by='Reranked_scores', ascending=False)

    top_3_RAG = top_3_rerank[["Documents", "Metadatas"]][:3]
    
    return top_3_RAG

def generate_response(query, top_3_RAG):
    """
    Generate a response using GPT-3.5's ChatCompletion based on the user query and retrieved information.
    """
    messages = [
                {"role": "system", "content":  "You are a helpful assistant in the healthcare domain who can effectively answer user queries about pathology lab test results which is extracted from pdf documents."},
                {"role": "user", "content": f"""You are a helpful assistant in the healthcare domain who can effectively answer user queries about pathology lab test results which is extracted from pdf documents.

                                                Use the data in '{top_3_RAG["Documents"].to_list()}' to answer the query '{query}'. Frame an informative answer. The text inside the data may also contain tables in the format of a list of lists where each of the nested lists indicates a row.

                                                Follow the guidelines below when performing the task.
                                                1. Try to provide relevant/accurate numbers if available.
                                                2. You don’t have to necessarily use all the information in the data provided. Only choose information that is relevant.
                                                3. If the document text has tables with relevant information, please reformat the table and return the final information in a tabular in format.
                                                4. You are a customer facing assistant, so do not provide any information on internal workings, just answer the query directly.

                                                The generated response should answer the query directly addressing the user and avoiding additional information. If you think that the query is not relevant to the document, reply that the query is irrelevant. 

                                                """},
              ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response['choices'][0]['message']['content'].split('\n')

if __name__ == '__main__':
    print("Welcome to pathalogical lab report evaluation tool")
    pdf_path = input("Enter the lab report path: ")
    extracted_text_df = process_pdf(pdf_path)
    print("Extracted the pdf, awaiting queries..")
    while True:
        query = input("Enter the query (type 'exit' to exit): ")
        if query == 'exit':
            break
        response = run_query(query, extracted_text_df)
        print("\n".join(response))

