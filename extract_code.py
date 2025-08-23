import ast
import sys

def extract_variable_string(source_file_path, variable_name):
    try:
        with open(source_file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        tree = ast.parse(source_code)

        for node in ast.walk(tree):
            # We are looking for an assignment statement
            if isinstance(node, ast.Assign):
                # Iterate through targets (in case of a, b = value)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == variable_name:
                        # Value can be ast.Constant (Python 3.8+) or ast.Str (older)
                        # ast.Constant.value for strings, ast.Str.s for strings
                        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                            print(node.value.value)
                            return True
                        elif isinstance(node.value, ast.Str): # For Python versions < 3.8
                            print(node.value.s)
                            return True
                        # Handle f-strings if necessary, though plugin_system_code is not one
                        # For f-strings, node.value would be ast.JoinedStr
                        # Extracting its exact original multiline string form can be tricky
                        # as ast normalizes it. For now, assume it's a simple string.

        # If variable not found or not a direct string assignment as expected
        print(f"Error: Variable '{variable_name}' not found as a direct string assignment.", file=sys.stderr)
        return False

    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_code.py <source_file_path> <variable_name>", file=sys.stderr)
        sys.exit(1) # Exit with 1 if wrong arguments

    source_file = sys.argv[1]
    var_name = sys.argv[2]

    if not extract_variable_string(source_file, var_name):
        sys.exit(1) # Exit with 1 if extraction function returns False
