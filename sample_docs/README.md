# Sample Financial Documents

This directory contains sample financial documents for testing the Fintech RAG API. These documents are designed to represent common financial document types that users might want to analyze using natural language queries.

## Available Documents

### 1. Bank Statement (`bank_statement.txt`)
- **Type**: Text file (TXT)
- **Content**: Monthly bank account statement for March 2024
- **Key Information**:
  - Account holder: John Smith
  - Account balance and transaction history
  - Fees and charges breakdown
  - Interest information
  - Contact details

**Sample Questions to Test**:
- "What is the current account balance?"
- "How much was paid in fees this month?"
- "What was the largest transaction?"
- "Who is the account holder?"
- "What is the interest rate?"

### 2. Credit Card Transactions (`credit_card_transactions.csv`)
- **Type**: CSV file
- **Content**: Credit card transaction data for March 2024
- **Key Information**:
  - 20 transactions across various categories
  - Merchant names, amounts, dates
  - Transaction categories (Entertainment, Food & Dining, etc.)
  - Late fees and interest charges

**Sample Questions to Test**:
- "What was the total spent on entertainment?"
- "Which merchant had the highest transaction amount?"
- "How many transactions were made at restaurants?"
- "What fees were charged this month?"
- "Show me all technology purchases"

### 3. Loan Agreement (`loan_agreement.txt`)
- **Type**: Text file (TXT)
- **Content**: Personal loan agreement document
- **Key Information**:
  - Borrower: Sarah Johnson
  - Loan amount: $15,000
  - Interest rate: 7.25% APR
  - Payment terms and conditions
  - Default consequences

**Sample Questions to Test**:
- "What is the monthly payment amount?"
- "What is the interest rate?"
- "What happens if the borrower defaults?"
- "How long is the loan term?"
- "What are the late payment fees?"

## Testing Instructions

1. **Upload Documents**: Use the `/upload` endpoint to upload each document
2. **Ask Questions**: Use the `/ask` endpoint with various natural language questions
3. **Test Scenarios**:
   - Document-specific queries (using document_id parameter)
   - Cross-document queries (without document_id)
   - Complex financial calculations
   - Policy and terms questions

## Expected API Responses

The RAG system should be able to:
- Extract specific numerical values (balances, payments, rates)
- Summarize document sections
- Compare information across documents
- Identify key terms and conditions
- Provide context-aware answers about financial products

## Note

These are sample documents created for testing purposes only. They do not represent real financial accounts, transactions, or agreements. 