import numpy as np
from src.evaluation.similarity import cosine_similarity, rouge_similarity, bert_similarity
from src.preprocessing import extract_text, clean_text
from src.embeddings import embed_texts
from src.vectorstore import create_index
from src.retriever import retrieve
from src.generator import generate_response
from src.compression import truncate_context
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks


def run_pipeline(pdf_path, query):

    print("Extracting text...")
    raw_text = extract_text(pdf_path)
    print("Raw length:", len(raw_text))

    print("Cleaning text for embedding...")
    cleaned_for_embed = clean_text(raw_text)

    print("Chunking...")
    chunks_embed = chunk_text(cleaned_for_embed)
    chunks_raw = chunk_text(raw_text)
    print("Total chunks:", len(chunks_embed))

    print("Embedding...")
    embeddings = embed_texts(chunks_embed)

    print("Creating FAISS index...")
    index = create_index(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    print("Retrieving relevant chunks...")
    retrieved = retrieve(query, index, chunks_raw)
    print("Retrieved chunks:", len(retrieved))

    context = " ".join(retrieved)
    context = truncate_context(context)

    print("Context words:", len(context.split()))

    # ---- PROMPT ----
    prompt = f"""
You are a technical research analyst.

Use ONLY the provided PATENT CONTEXT.
Do NOT copy large verbatim text.
Do NOT fabricate unrelated inventions.

PATENT CONTEXT:
{context}

Generate the following sections clearly and completely:

Technical Summary
Core Innovation
Implementation Details
Impact & Applications

"""

    print("Generating response...")
    response = generate_response(prompt)

    print("\n===== MODEL OUTPUT =====\n")
    print(response)

    # ---- EVALUATION ----
    gold_output = """
Innovation
The patent introduces a system and method for collaborative notifications in healthcare settings, with the following key innovations:
Collaborative Healthcare System: Facilitates communication among care providers and virtual healthcare assistants (VHAs) through patient-specific communication channels, including communication threads and dashboards.
Virtual Healthcare Assistants (VHAs): AI-driven assistants that retrieve patient information, provide care guidelines, predict future patient states, and monitor patient conditions. They understand natural language inputs from care providers.
Patient-Specific Communication Channels: Each patient has a dedicated communication channel with a communication thread for messaging and a dashboard for displaying patient-specific medical information.
Real-Time Monitoring and Alerts: Monitors patient vital signs and other medical data in real-time, generating alerts and notifications when certain conditions are met.
Personalized Notifications: Care providers can set personalized notifications for specific patient conditions or reminders, delivered in a timely manner to ensure prompt action.
Timeline Feature: Provides a summary of relevant patient medical events over a predefined period, facilitating shift handovers and quick reviews of patient status.
Integration with External Services: Integrates with external guideline and prediction services to provide up-to-date care guidelines and predictive analytics.

Abstract
The patent describes systems and methods for collaborative notifications in healthcare environments. The system includes a display and a computing device that generates and outputs a patient-specific communication thread. This thread includes communication among care providers and a virtual healthcare assistant. The system generates notifications indicating changes in a patient's state and outputs these notifications to the display. The notifications are integrated into the communication thread and can be viewed as part of the patient's communication history. The system aims to improve communication among care providers, reduce workload, and enhance patient care through real-time monitoring and personalized alerts.

Methodology
The methodology involves several key steps and components:
System Architecture: The system includes a collaborative space server system that stores and executes communication threads, dashboards, and digital twins for each patient. It is connected to hospital operational systems, monitoring devices, and external services.
Communication Threads and Dashboards: Each patient has a dedicated communication channel with a communication thread and a dashboard. The communication thread facilitates text and rich-media-based messages among care providers and VHAs, while the dashboard displays patient-specific medical information.


Virtual Healthcare Assistants (VHAs):

EMR VHA: Retrieves patient information from electronic medical records.
Guideline VHA: Retrieves care guidelines from external services.
Predictive VHA: Predicts future patient states using external prediction services.
Listening VHA: Monitors communication and patient surroundings to infer patient status.
Monitoring VHA: Receives and tracks patient data from monitoring devices.
Notifications: The system generates and outputs notifications based on patient monitoring data, user requests, or changes in patient state. Notifications are classified by importance and can be personalized.
Timeline Generation: The system generates a timeline of relevant patient medical events over a predefined period, which can be filtered and viewed by care providers.
User Interaction: Care providers interact with the system through natural language inputs, selecting messages, and setting notification preferences.


Results
The results of implementing the system described in the patent include:
Improved Communication: Enhanced communication and collaboration among care providers through dedicated communication channels and real-time updates.
Reduced Workload: Virtual healthcare assistants handle information retrieval, patient monitoring, and predictive tasks, reducing the workload on care providers.
Timely Alerts and Notifications: Personalized and automatic notifications ensure that care providers are promptly informed of changes in patient states, facilitating timely interventions.
Enhanced Patient Care: Real-time monitoring, predictive analytics, and access to up-to-date care guidelines contribute to improved patient outcomes.
Efficient Handover: The timeline feature provides a concise summary of patient events, making shift handovers more efficient and reducing the risk of information loss.
Customizable and User-Friendly Interface: The system offers a customizable and intuitive interface, allowing care providers to easily access relevant patient information and set personalized notifications.




"""

    cos_sim = cosine_similarity(response, gold_output)
    rouge_sim = rouge_similarity(response, gold_output)
    bert_sim = bert_similarity(response, gold_output)

    print("\n===== SIMILARITY SCORES =====")
    print(f"Cosine: {cos_sim}")
    print(f"Rouge: {rouge_sim}")
    print(f"BERT: {bert_sim}")

    return response


# After the above sections, provide evaluation strictly in valid JSON format.

# Each JSON text field must contain exactly 2 concise sentences.
# Scores must be integers between 0 and 10.

# Return ONLY the sections followed by JSON.

# {{
#     "novelty_score": 0,
#     "inventive_step_score": 0,
#     "overlap_analysis": "",
#     "key_differences": "",
#     "technical_insights": "",
#     "future_research_direction": ""
# }}
