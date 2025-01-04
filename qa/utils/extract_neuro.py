import requests
import torch
from torch_geometric.data import Data
import xml.etree.ElementTree as ET
import time


# Base URL for the API
base_url = "http://cng.gmu.edu:8080/api"
page_size = 500  # Maximum allowed page size
neuron_data_with_pmid = []

# Function to get all neurons with a reference_pmid
def get_all_neurons_with_pmid():
    page = 0  # Start from page 0
    while True:
        # Endpoint for fetching neurons with paging
        url = f"{base_url}/neuron?page={page}&size={page_size}&sort=neuron_id,asc"
        response = requests.get(url)

        # Check if the response is successful
        if response.status_code == 200:
            data = response.json()
            neurons = data["_embedded"]["neuronResources"]

            # Filter neurons with a reference_pmid
            for neuron in neurons:
                reference_pmid = neuron.get("reference_pmid")
                if reference_pmid:
                    neuron_data_with_pmid.append({
                        "neuron_id": neuron["neuron_id"],
                        "neuron_name": neuron["neuron_name"],
                        "reference_pmid": reference_pmid
                    })

            # Move to the next page
            if "next" in data["_links"]:
                page += 1
            else:
                break  # Exit loop if there's no next page
        else:
            print(f"Failed to fetch data at page {page}, status code: {response.status_code}")
            break

    return neuron_data_with_pmid


# Get detailed information of one neuron in the database.
def get_neuron_info(neuron_id):
    base_url = "http://cng.gmu.edu:8080/api"
    url = f"{base_url}/neuron/id/{neuron_id}"

    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "status_code": response.status_code,
            "error": "Failed to fetch neuron information"
        }
    

# Get all literatures in the database
def get_all_literatures():
    base_url = "http://cng.gmu.edu:8080/api"
    url = f"{base_url}/literature"
    all_literatures = []
    page = 0
    page_size = 500  # Adjust to a maximum of 500 if desired

    while True:
        # Request data for the current page
        response = requests.get(url, params={"page": page, "size": page_size})

        if response.status_code == 200:
            data = response.json()

            # Check if '_embedded' and 'publicationResources' exist in response
            if "_embedded" in data and "publicationResources" in data["_embedded"]:
                publications = data["_embedded"]["publicationResources"]
                all_literatures.extend(publications)
            else:
                print(f"No publication data found on page {page}.")
                break

            # Check if there is a 'next' page; if not, exit loop
            if "next" in data["_links"]:
                page += 1
            else:
                break
        else:
            print(f"Error: Received status code {response.status_code}")
            break

    return all_literatures


# Extract article_id and pmid pairs
def extract_id_pairs(literatures):
    id_pairs = []
    for publication in literatures:
        article_id = publication.get("article_id")
        pmid = publication.get("pmid")
        # doi = publication.get("doi")
        if article_id and pmid:
            id_pairs.append({"article_id": article_id, "pmid": pmid})
    return id_pairs


def get_references(pubmed_id):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&id={pubmed_id}&linkname=pubmed_pubmed_refs&retmode=json"
    response = requests.get(url).json()
    references = response["linksets"][0]["linksetdbs"][0]["links"]
    return references

def fetch_title_abstract(pubmed_id, max_retries=3, delay=1):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pubmed_id}&retmode=xml"
    for attempt in range(max_retries):
        response = requests.get(url)

        # Check for Too Many Requests status
        if response.status_code == 429:
            print(f"Rate limit hit for PubMed ID {pubmed_id}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
            continue  # Retry the request

        # Check if the response is successful
        if response.status_code == 200:
            try:
                # Parse the XML response
                root = ET.fromstring(response.content)
                title = root.findtext(".//ArticleTitle") or "No Title"
                abstract = " ".join([t.text for t in root.findall(".//AbstractText") if t.text]) or "No Abstract"
                return title, abstract
            except ET.ParseError as e:
                print(f"XML parsing error for PubMed ID {pubmed_id}: {e}")
                return "No Title", "No Abstract"

        print(f"Failed to fetch data for PubMed ID {pubmed_id}. Status code: {response.status_code}")
        return "No Title", "No Abstract"

    print(f"Max retries exceeded for PubMed ID {pubmed_id}.")
    return "No Title", "No Abstract"


def build_tree(pubmed_id):
    # Get references
    references = get_references(pubmed_id)

    # Initialize lists to store node features and edges
    titles = []
    abstracts = []
    edge_index = []

    # Fetch title and abstract for the central node
    title, abstract = fetch_title_abstract(pubmed_id)
    titles.append(title)
    abstracts.append(abstract)

    # Process each reference (1-hop neighbor)
    for i, ref_id in enumerate(references, start=1):
        ref_title, ref_abstract = fetch_title_abstract(ref_id)

        # Add reference titles and abstracts
        titles.append(ref_title)
        abstracts.append(ref_abstract)

        # Create edges (central node <-> reference)
        edge_index.append([0, i])  # central -> reference
        # edge_index.append([i, 0])  # reference -> central

    # Convert edge_index to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create and return the PyG data object with raw text as attributes
    data = Data(edge_index=edge_index)
    data.titles = titles
    data.abstracts = abstracts
    return data


def build_graph(pubmed_id):
    # Initialize data structures for nodes and edges
    titles = []
    abstracts = []
    edge_index = []
    node_map = {pubmed_id: 0}  # Map PubMed ID to node index
    current_index = 1  # Next node index

    # Central node
    title, abstract = fetch_title_abstract(pubmed_id)
    titles.append(title)
    abstracts.append(abstract)

    # Get references of the central node
    references = get_references(pubmed_id)

    # Process each reference (1-hop neighbor)
    for ref_id in references:
        # Check if the reference already exists in the node map
        if ref_id not in node_map:
            ref_title, ref_abstract = fetch_title_abstract(ref_id)
            titles.append(ref_title)
            abstracts.append(ref_abstract)
            node_map[ref_id] = current_index
            current_index += 1
        # Add edge from central node to reference
        edge_index.append([node_map[pubmed_id], node_map[ref_id]])

    # Process cross-citations between references
    for ref_id in references:
        ref_index = node_map[ref_id]
        ref_references = get_references(ref_id)
        for cross_ref_id in ref_references:
            if cross_ref_id in node_map:
                cross_ref_index = node_map[cross_ref_id]
                edge_index.append([ref_index, cross_ref_index])

    # Convert edge_index to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create and return the PyG data object with raw text attributes
    data = Data(edge_index=edge_index)
    data.titles = titles
    data.abstracts = abstracts
    return data
