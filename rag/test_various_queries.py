#!/usr/bin/env python3
"""
Test script to validate the RAG system with various constitutional questions
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval_pipeline import retrieve_and_answer

# Test queries covering different topics
test_queries = [
    "How is the Prime Minister elected in Nepal?",
    "What are the fundamental rights of citizens in Nepal?",
    "What are the duties of citizens?",
    "How is the President elected?",
    "What is the structure of the Federal Parliament?",
    "What are the provisions for freedom of speech?",
]

print("="*70)
print("TESTING RAG SYSTEM WITH VARIOUS CONSTITUTIONAL QUESTIONS")
print("="*70)

for i, query in enumerate(test_queries, 1):
    print(f"\n\n{'#'*70}")
    print(f"TEST {i}/{len(test_queries)}")
    print(f"{'#'*70}\n")
    
    try:
        retrieve_and_answer(query, verbose=False)
    except Exception as e:
        print(f"ERROR: {e}")
    
    print("\n" + "-"*70)

print("\n\n" + "="*70)
print("ALL TESTS COMPLETED")
print("="*70)
