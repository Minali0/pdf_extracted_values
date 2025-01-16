import pandas as pd

# Load the Excel file
file_path = '/home/seaflux/Documents/Document-Query-Q-A_Quotey/Benefits cover table - 12.19.24.xlsx'  # Replace with your actual file path
excel_data = pd.ExcelFile(file_path)

# Load data from the first sheet
sheet_data = excel_data.parse('Benefits cover table - 12.19.24')

# Select only the first and second columns
sheet_data = sheet_data.iloc[:, :2]  # This will select all rows and the first two columns

# Rename columns to "ID" and "Query"
sheet_data.rename(columns={
    sheet_data.columns[0]: 'ID', 
    sheet_data.columns[1]: 'Query'
}, inplace=True)

# Drop rows where any column has a blank or NaN value
sheet_data.dropna(subset=['ID', 'Query'], inplace=True)

# Generate questions based on the "Query" column content
sheet_data['Query'] = sheet_data['Query'].apply(lambda x: f"What is the {x.lower()}?")

# Save the updated DataFrame to a new CSV file
output_path = 'questions_generated.csv'  # Specify your desired output path
sheet_data.to_csv(output_path, index=False)

print(f"The updated file has been saved to {output_path}")
