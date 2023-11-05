# a python script to comapre 2x requiremnts.txt and seperate the differnces

def remove_common_entries(requirements_path, eal_base_path, output_path):
    with open(requirements_path, 'r') as file:
        requirements_content = set(file.readlines())

    with open(eal_base_path, 'r') as file:
        eal_base_content = set(file.readlines())

    # Remove common entries
    updated_requirements = requirements_content - eal_base_content

    with open(output_path, 'w') as file:
        for line in updated_requirements:
            file.write(line)

# Define the paths to the input files and the output file
requirements_path = 'requirements2.txt'
eal_base_path = 'requirements.txt'
output_path = 'updated_requirements.txt'

# Call the function to remove common entries and write the updated requirements to the output file
remove_common_entries(requirements_path, eal_base_path, output_path)
