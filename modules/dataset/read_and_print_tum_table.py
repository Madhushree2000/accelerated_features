def read_and_print_tum_table(filepath):
    """
    Reads a .tum file containing a table of numbers separated by spaces or tabs
    and prints each line.
    """
    try:
        with open(filepath, 'r') as file:
            print(f"--- Contents of '{filepath}' ---")
            for line_num, line in enumerate(file, 1):
                # .strip() removes leading/trailing whitespace (like newlines)
                # You can then split the line into individual numbers if needed
                numbers_as_strings = line.strip().split()
                print(f"Line {line_num}: {numbers_as_strings}")
                # If you want to convert them to actual numbers (e.g., floats):
                # try:
                #     numbers = [float(num) for num in numbers_as_strings]
                #     print(f"Line {line_num} (as floats): {numbers}")
                # except ValueError:
                #     print(f"Could not convert all values on line {line_num} to numbers.")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Replace 'your_table.tum' with the actual path to your .tum file
    tum_file_path = "/Volumes/Lenovo PS6/Dataset/fjord_3_baseline.tum"
    read_and_print_tum_table(tum_file_path)