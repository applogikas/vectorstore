# RAG (Retrieval-Augmented Generation) with Milvus

This tutorial demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline using Milvus and OpenAI. The system retrieves relevant documents from a corpus using Milvus and generates new text based on these retrieved documents using a generative model.

## Prerequisites

Before starting, ensure that you have the following installed:

- Python 3.13.1
- pip (Python package installer)
- OpenAI API key
- Milvus installed (or use Milvus Lite)

## Step-by-Step Guide

### 1. Install Dependencies

First, you need to install the required packages. Run the following command to install the necessary Python libraries:

```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key

You need an OpenAI API key to interact with the OpenAI models. Set up your OpenAI API key as an environment variable by adding it to a `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key
```

### 3. Set Up Milvus Database

You also need to set up the Milvus database connection. Add the following details to your `.env` file:

```env
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=root:Milvus
```

### How to Run:

1. Install the dependencies using the command provided.
2. Set your OpenAI API key and put it into the `.env` file (rename `example.env` to `.env`).
3. Follow the steps outlined in the guide to load the data, create embeddings, insert them into Milvus, and build the RAG pipeline.
4. Retrieve answers to queries using the RAG system.
