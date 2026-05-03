from data.preprocessing import get_data_loaders

data_path = "HAM10000"

outputs = get_data_loaders(data_path)

print("Preprocessing works!")
print("Number of returned outputs:", len(outputs))

for i, output in enumerate(outputs):
    print(f"Output {i}:", type(output))

    